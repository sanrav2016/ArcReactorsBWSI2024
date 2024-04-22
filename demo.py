#!/usr/bin/python
# -*- coding: UTF-8 -*-
#import chardet

import dlib
import cv2
import argparse, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from model import model_static
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from colour import Color
import mediapipe as mp
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2, Preview
import sys 
import logging
import spidev as SPI
sys.path.append("../lg-master/LCD_Module_RPI_code/RaspberryPi/python")
from lib import LCD_1inch28

GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.OUT)

parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
parser.add_argument('--face', type=str, help='face detection file path. dlib face detector is used when not specified')
parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
parser.add_argument('--jitter', type=int, help='jitter bbox n times, and average results', default=0)
parser.add_argument('-save_vis', help='saves output as video', action='store_true')
parser.add_argument('-save_text', help='saves output as text', action='store_true')
parser.add_argument('-display_off', help='do not display frames', action='store_true')

args = parser.parse_args()

CNN_FACE_MODEL = 'data/mmod_human_face_detector.dat' # from http://dlib.net/files/mmod_human_face_detector.dat.bz2

RST = 27
DC = 25
BL = 18
bus = 0 
device = 0 
logging.basicConfig(level=logging.DEBUG)

def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right+bbox_left)/2.0
    cy = (bbox_bottom+bbox_top)/2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right-cx)*scale + cx
    bbox_left = (bbox_left-cx)*scale + cx
    bbox_top = (bbox_top-cy)*scale + cy
    bbox_bottom = (bbox_bottom-cy)*scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def run(video_path, face_path, model_weight, jitter, vis, display_off, save_text):
    # set up vis settings
    red = Color("red")
    colors = list(red.range_to(Color("green"),10))
    font = ImageFont.truetype("data/arial.ttf", 40)

    w, h = 640, 480

    # set up video source
    if video_path is None:
        picam2 = Picamera2()
        picam2.start()
        #cap = cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1);
        video_path = 'live.avi'
    else:
        cap = cv2.VideoCapture(video_path)
        
    vid1_src = "eye1.mp4"
    cap1 = cv2.VideoCapture(vid1_src)
    vid2_src = "eye2.mp4"
    cap2 = cv2.VideoCapture(vid2_src)
        
        #disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, device),spi_freq=10000000,rst=RST,dc=DC,bl=BL)
    disp = LCD_1inch28.LCD_1inch28()
    # Initialize library.
    disp.Init()
    # Clear display.
    disp.clear()
    #Set the backlight to 100
    disp.bl_DutyCycle(50)

    # set up output file
    if save_text:
        outtext_name = os.path.basename(video_path).replace('.avi','_output.txt')
        f = open(outtext_name, "w")
    if vis:
        outvis_name = os.path.basename(video_path).replace('.avi','_output.avi')
        imwidth = int(w); imheight = int(h)
        outvid = cv2.VideoWriter(outvis_name,cv2.VideoWriter_fourcc('M','J','P','G'), 15, (imwidth,imheight))

    # set up face detection mode
    if face_path is None:
        facemode = 'DLIB'
    else:
        facemode = 'GIVEN'
        column_names = ['frame', 'left', 'top', 'right', 'bottom']
        df = pd.read_csv(face_path, names=column_names, index_col=0)
        df['left'] -= (df['right']-df['left'])*0.2
        df['right'] += (df['right']-df['left'])*0.2
        df['top'] -= (df['bottom']-df['top'])*0.1
        df['bottom'] += (df['bottom']-df['top'])*0.1
        df['left'] = df['left'].astype('int')
        df['top'] = df['top'].astype('int')
        df['right'] = df['right'].astype('int')
        df['bottom'] = df['bottom'].astype('int')

    #if (cap.isOpened()== False):
    #    print("Error opening video stream or file")
    #    exit()

    if facemode == 'DLIB':
        mp_face_detection = mp.solutions.face_detection
        cnn_face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        #cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)
    frame_cnt = 0

    # set up data transformation
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load model weights
    model = model_static(model_weight)
    model_dict = model.state_dict()
    snapshot = torch.load(model_weight, map_location=torch.device("cpu"))
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    model.cpu()
    model.train(False)

    frame_rate = 10
    prev = 0
    
    activate_flag = False

    # video reading loop
    while True:
    #while (cap.isOpened()):
        ret1, frame1 = cap1.read()
        if ret1:
            image1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            disp.ShowImage(image1)
        else: 
            cap1 = cv2.VideoCapture(vid1_src)
    
        ret = True
        frame = picam2.capture_array()
        #ret, frame = cap.read() 
        time_elapsed = time.time() - prev
        if time_elapsed < 1./frame_rate:
            continue
        prev = time.time()
        if ret == True:
            height, width, channels = frame.shape
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_cnt += 1
            bbox = []
            if facemode == 'DLIB':
                dets = cnn_face_detector.process(frame)
                if dets.detections:
                    for d in dets.detections:
                        rect = d.location_data.relative_bounding_box
                        l = rect.xmin * w#d.rect.left()
                        t = rect.ymin * h#d.rect.top()
                        r = l + rect.width * w#d.rect.right()
                        b = t + rect.height * h#d.rect.bottom()
                        # expand a bit
                        l -= (r-l)*0.2
                        r += (r-l)*0.2
                        t -= (b-t)*0.2
                        b += (b-t)*0.2
                        bbox.append([l,t,r,b])
            elif facemode == 'GIVEN':
                if frame_cnt in df.index:
                    bbox.append([df.loc[frame_cnt,'left'],df.loc[frame_cnt,'top'],df.loc[frame_cnt,'right'],df.loc[frame_cnt,'bottom']])

            frame = Image.fromarray(frame)
            for b in bbox:
                face = frame.crop((b))
                img = test_transforms(face)
                img.unsqueeze_(0)
                if jitter > 0:
                    for i in range(jitter):
                        bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
                        bj = [bj_left, bj_top, bj_right, bj_bottom]
                        facej = frame.crop((bj))
                        img_jittered = test_transforms(facej)
                        img_jittered.unsqueeze_(0)
                        img = torch.cat([img, img_jittered])

                # forward pass
                output = model(img.cpu())
                if jitter > 0:
                    output = torch.mean(output, 0)
                score = F.sigmoid(output).item()
                
                coloridx = min(int(round(score*10)),9)
                draw = ImageDraw.Draw(frame)
                drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
                t = str(round(score,2))
                draw.text((b[0],b[3]), t, fill=(255,255,255,128), font=font)
                if float(t) >= 0.85:
                    if not activate_flag:
                        activate_flag = True
                        GPIO.output(2, GPIO.HIGH)
                        time.sleep(0.1)
                        GPIO.output(2, GPIO.LOW)
                        count = 0
                        while count < 2:
                            ret2, frame2 = cap2.read()
                            if ret2:
                                image2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
                                disp.ShowImage(image2)
                            else: 
                                cap2 = cv2.VideoCapture(vid2_src)
                                count += 1
                else:
                    if activate_flag:
                        activate_flag = False
                if save_text:
                    f.write("%d,%f\n"%(frame_cnt,score))

            if not display_off:
                frame = np.asarray(frame) # convert PIL image back to opencv format for faster display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #cv2.imshow('',frame)
                if vis:
                    outvid.write(frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        else:
            break

    if vis:
        outvid.release()
    if save_text:
        f.close()
    cap.release()
    print('DONE!')


if __name__ == "__main__":
    run(args.video, args.face, args.model_weight, args.jitter, args.save_vis, args.display_off, args.save_text)
