import tensorflow as tf
import numpy as np
import os 
import cv2
from tensorflow.keras.models import Sequential, save_model, load_model

classes = ["Pick","Stitch","Person"]
positions = ["Left"]

n_frames = 8
step = 1 
interval = [2,3,4]
boxes = [(112,259,152,152)]

path_model = "3D CNN\\Checkpoint" 
path_input = "RepNet-Pytorch-main\\stitching-videos"
path_output = "3D CNN\\Stitching Prediction"
filenames = ["output_1.mp4","output_3.mp4","vlc-record-2021-12-22-18h56m13s-Converting rtsp___admin_Admin@123!@103.210.28.115_1070_live2.sdp-.mp4"]

model = load_model(f'{path_model}\\model', compile = True)

def preprocess(frame) : 
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame,(64,64))
    frame = frame/255
    return frame


def predict_clip(frames_list) :

    global n_frames

    clips = []

    for frames in frames_list : 

        total_frames = len(frames)

        frames_data = []

        for i in range(1,n_frames+1) : 
            frames_data.append(frames[int((i/(n_frames+1))*total_frames)])

        clips.append(frames_data)

    clips = np.array(clips,dtype=np.float32)
    clips = clips.reshape(clips.shape[0], 8, 64, 64, 1)

    predictions = model.predict(clips)

    predict_classes = []

    for i in predictions : 
        predict_classes.append(classes[np.argmax(i)])

    Person = predict_classes.count("Person")
    Pick = predict_classes.count("Pick")
    Stitch = predict_classes.count("Stitch")

    if(Person>=Pick and Person>=Stitch) : return "Person"
    if(Pick>=Stitch) : return "Pick"
    return "Stitch"
    

def run_model(box,path,s) :

    video = cv2.VideoCapture(path)
    
    fps = int(video.get(cv2.CAP_PROP_FPS))
    x,y,w,h = box
    ret,src = video.read()

    result = cv2.VideoWriter(f'{path_output}\\{s[:-4]}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(800,600))

    frames_list = []
    for i in range(len(interval)) : frames_list.append([])

    ct = 0

    prediction = ""
    pick_count = 0 
    pick_diff = fps*15

    while ret:
        for i in range(len(interval)) :
            frames_list[i].append(preprocess(src[y:y+h,x:x+w]))
            if(len(frames_list[i])>(interval[i]*fps)) : del frames_list[i][0]
        ct = ct + 1
        pick_diff = pick_diff + 1
        if(ct%(int(fps*step))==0 and ct>=(max(interval)*fps)) : prediction = predict_clip(frames_list)
        if(prediction=="Pick") : 
            if(pick_diff>fps*6) :
                pick_count = pick_count + 1
                pick_diff = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(src,prediction,(250,570), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(src,f"Pick Count : {pick_count}",(300,100), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        result.write(src) 
        ret,src = video.read()
    video.release()
    result.release()


for file in os.listdir(path_input) :
    
    if file in filenames : 
        path=os.path.join(f"{path_input}\\",file)
        run_model(boxes[0],path,file)
        print(f"Done : {file}")

