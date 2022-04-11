import cv2
import numpy as np
import time
import random
import os
import numpy as np
import json
import matplotlib.pyplot as plt

# EDIT
path_input = "RepNet-Pytorch-main\\stitching-videos"
path_output = "Repetition_Plots"
boxes = [(197,295,66,111),(460,282,63,129),(696,314,51,62)]
filenames = ["output_1.mp4","output_3.mp4","vlc-record-2021-12-22-18h56m13s-Converting rtsp___admin_Admin@123!@103.210.28.115_1070_live2.sdp-.mp4"]
# EDIT 

def find_peaks(path,s) :

    for count,box in enumerate(boxes) :

        video = cv2.VideoCapture(path)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        x,y,w,h = box
        ret,src = video.read()

        n_frames = 0
        points = []

        while ret:

            img = src[y:y+h,x:x+w]
            img = cv2.Canny(image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), threshold1=10, threshold2=20)
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
            points.append(int(np.mean(img)))
            n_frames = n_frames + 1
            #if(n_frames%1000==0) : print(n_frames)
            ret,src = video.read() 

        video.release()

        y = list(range(n_frames))
        for i in range(len(y)) : y[i] = y[i]/(fps*60)
        plt.scatter(y, points,s=1)
        plt.savefig(f'{path_output}\\{s[:-4]}_{count}.png')
        with open(f'{path_output}\\{s[:-4]}_{count}.json', 'w') as fp:
            json.dump(points, fp)
        plt.clf()


for file in os.listdir(path_input) :
    
    if file in filenames : 
        path=os.path.join(f"{path_input}\\",file)
        find_peaks(path,file)
        print(f"Done : {file}")

