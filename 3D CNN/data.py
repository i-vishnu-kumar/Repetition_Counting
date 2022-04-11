import cv2
import numpy as np
import time
import random
import os
import numpy as np
import json
import matplotlib.pyplot as plt

# EDIT
path_in = "3D CNN\\Data\\Videos"
path_out = "3D CNN\\Data"
classes = ["Person"]
positions = ["Left"]
n_frames = 8
boxes = [(112,259,152,152)]
# EDIT 

n = 0

def preprocess(frame) : 
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame,(64,64))
    frame = frame/255
    return frame

def make_data(box,path,path_output) :

    global n,n_frames

    video = cv2.VideoCapture(path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    x,y,w,h = box
    ret,src = video.read()

    frames = []

    while ret:
        frames.append(preprocess(src[y:y+h,x:x+w]))
        ret,src = video.read()
    video.release()

    total_frames = len(frames)

    frames_data = []

    for i in range(1,n_frames+1) : 
        frames_data.append(frames[int((i/(n_frames+1))*total_frames)])
    #with open(f'{path_output}\\{n}.json', 'w') as fp: json.dump(frames_data, fp)
    np.save(f'{path_output}\\{n}.npy',frames_data)
    n = n + 1
    for j in range(3) : 
        frames_data = []
        high = 0
        for i in range(1,n_frames+1) :
            low = high
            high = int((i/(n_frames))*total_frames)-1
            frames_data.append(frames[random.randint(low,high)])
        np.save(f'{path_output}\\{n}.npy',frames_data)
        n = n + 1


for cls in classes : 
    for pos_no,pos in enumerate(positions) : 
        n = 0
        for file in os.listdir(f"{path_in}\\{cls}\\{pos}") :
            print(file)
            make_data(boxes[pos_no],os.path.join(f"{path_in}\\{cls}\\{pos}\\",file),f"{path_out}\\{cls}\\{pos}")

