import cv2
import numpy as np
import time
import random
import os
postfix = 1

# EDIT
Dataset_Path = "QUVA\\videos"
Output_Path = "Augmented_Data_2"
# EDIT


'''
Same Video
'''
def A00(video,s) :

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()
    while ret:
        result.write(src) 
        ret,src = video.read() 

    result.release() 


'''
Mirror
'''
def A01(video,s) :

    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   
    
    ret,src = video.read()

    while ret:
        # EDIT
        src = src[:, ::-1, :]
        result.write(src) 
        # EDIT
        ret,src = video.read() 
    
    result.release()


'''
Invert
'''
def A02(video,s) : 

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:
        # EDIT
        src = src[::-1, :, :]
        result.write(src) 
        # EDIT
        ret,src = video.read() 
    
    result.release()


'''
Mirror + Invert
'''
def A03(video,s) : 

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:
        # EDIT
        src = src[::-1, ::-1, :]
        result.write(src) 
        # EDIT
        ret,src = video.read()  

    result.release()


'''
Camera Shaking
'''
def A11(video,s) :

    # EDIT
    ct = 0
    ct_add = True
    angle_change = 1
    angle = 0
    neg = False
    pos = False
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        center = (src.shape[1]//2, src.shape[0]//2)
        angle_change = random.uniform(-5, 5)
        if(angle>45) : neg = True
        if(angle<-45) : pos = True
        if(neg) : 
            angle_change = random.uniform(-5,0)
            if(angle<20) : neg = False
        if(pos) : 
            angle_change = random.uniform(0,5)
            if(angle>-20) : pos = False
        angle = angle + angle_change
        scale = 1
        rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
        warp_rotate_dst = cv2.warpAffine(src, rot_mat, (src.shape[1], src.shape[0]))
        result.write(warp_rotate_dst) 
        # EDIT
        ret,src = video.read() 

    result.release()


'''
Camera Shaking + Scale Changing
'''
def A12(video,s) : 

    # EDIT
    ct = 0
    ct_add = True
    angle_change = 1
    angle = 0
    scale = 1
    scale_change = 0
    neg = False
    pos = False
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        center = (src.shape[1]//2, src.shape[0]//2)
        angle_change = random.uniform(-5, 5)
        if(angle>45) : neg = True
        if(angle<-45) : pos = True
        if(neg) : 
            angle_change = random.uniform(-5,0)
            if(angle<20) : neg = False
        if(pos) : 
            angle_change = random.uniform(0,5)
            if(angle>-20) : pos = False
        angle = angle + angle_change
        scale_low = 0.6
        scale_up = 1
        scale_change = random.uniform(-0.1,0.1)
        if(scale<(scale_low+0.1)) : scale_change = random.uniform(0,0.1)
        if(scale>(scale_up-0.1)) : scale_change = random.uniform(-0.1,0)
        scale = scale + scale_change
        if ct_add : ct = ct + 1
        else : ct = ct - 1
        rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
        warp_rotate_dst = cv2.warpAffine(src, rot_mat, (src.shape[1], src.shape[0]))
        result.write(warp_rotate_dst) 
        # EDIT
        ret,src = video.read() 

    result.release()


'''
Continuous Rotation
'''
def A13(video,s) :

    # EDIT
    ct = 0
    ct_add = True
    angle_change = 1
    angle = 0
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        center = (src.shape[1]//2, src.shape[0]//2)
        angle_change = random.uniform(0,1)
        angle = angle + angle_change
        scale = 1
        rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
        warp_rotate_dst = cv2.warpAffine(src, rot_mat, (src.shape[1], src.shape[0]))
        result.write(warp_rotate_dst)
        # EDIT 
        ret,src = video.read() 

    result.release()


'''
Warping 
'''
def A21(video,s) :

    # EDIT
    ct = 0
    ct_add = True
    angle_change = 1
    angle = 0
    c1 = 0.33
    c2 = 0.85
    c3 = 0.25
    c4 = 0.15
    c5 = 0.7
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
        v1 = random.uniform(-0.1,0.1)
        v2 = random.uniform(-0.1,0.1)
        v3 = random.uniform(-0.1,0.1)
        v4 = random.uniform(-0.1,0.1)
        v5 = random.uniform(-0.1,0.1)
        if(c1>0.4) : c1 = c1-abs(v1)
        elif(c1<0.1) : c1 = c1 + abs(v1)
        else : c1 = c1 + v1
        if(c2>0.9) : c2 = c2-abs(v2)
        elif(c2<0.7) : c2 = c2 + abs(v2)
        else : c2 = c2 + v2
        if(c3>0.9) : c3 = c3-abs(v3)
        elif(c3<0.5) : c3 = c3 + abs(v3)
        else : c3 = c3 + v3
        if(c4>0.9) : c4 = c4-abs(v4)
        elif(c4<0.5) : c4 = c4 + abs(v4)
        else : c4 = c4 + v4
        if(c5>0.9) : c5 = c5-abs(v5)
        elif(c5<0.9) : c5 = c5 + abs(v5)
        else : c5 = c5 + v5
        dstTri = np.array( [[0, src.shape[1]*c1], [src.shape[1]*c2, src.shape[0]*0.25], [src.shape[1]*0.15, src.shape[0]*c5]] ).astype(np.float32)
        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
        result.write(warp_dst) 
        # EDIT
        ret,src = video.read() 

    result.release()


'''
Image Addition
'''
def A31(video,s) :

    # EDIT
    weight_img = 0.3
    img = cv2.imread("Boxes_Stitching.jpg")
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    img = cv2.resize(img,(src.shape[1],src.shape[0]))

    while ret:

        # EDIT
        result.write(cv2.addWeighted(img,weight_img,src,1-weight_img,0))
        # EDIT 
        ret,src = video.read() 

    result.release()


'''
Noise Addition
'''
def A32(video,s) :

    # EDIT
    noise_min = 0
    noise_max = 40
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    f = noise_max/255

    while ret:

        # EDIT
        noise = np.random.randint(int(255*noise_min/noise_max),255,size=(src.shape[0],src.shape[1],3), dtype=np.uint8)
        result.write(cv2.addWeighted(noise,f,src,1-f,0))
        # EDIT 
        ret,src = video.read() 

    result.release()


'''
Swappping Color Channels
'''
def A41(video,s) :

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        rgb = [src[:,:,2].copy(),src[:,:,1].copy(),src[:,:,0].copy()]
        src[:,:,0] = rgb[random.randint(0,2)]
        src[:,:,1] = rgb[random.randint(0,2)]
        src[:,:,2] = rgb[random.randint(0,2)]
        result.write(src)
        # EDIT 
        ret,src = video.read() 

    result.release()


'''
Speed Change
'''
def A51(video,s) :

    # EDIT
    speed = 2
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),int(fps*speed),(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        result.write(src)
        # EDIT 
        ret,src = video.read() 

    result.release()


'''
Skipping Frames
'''
def A52(video,s) :

    # EDIT
    skip = 3
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),int(fps),(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        p = random.randint(1,skip)
        if(p!=skip) : result.write(src)
        # EDIT 
        ret,src = video.read() 

    result.release()


'''
Combine n continuous Frames
'''
def A61(video,s) :

    # EDIT
    n = 4
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),int(fps),(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    prev = []

    while ret:

        # EDIT 
        prev.append(src)
        if(len(prev)>=n) :
            c = 2
            for fr in prev[0:-1] : 
                w = 1/c
                src = cv2.addWeighted(src,1-w,fr,w,0)
                c = c + 1
            del prev[0]
            result.write(src)
        # EDIT 
        ret,src = video.read() 

    result.release()

#========================================================================================

# EDIT
Augment = [A52]
# EDIT

coun = 0

for file in os.listdir(Dataset_Path) :
 

    postfix = 15

    path=os.path.join(f"{Dataset_Path}\\", file)
    s = str(file)

    for aug in Augment :

        video = cv2.VideoCapture(path)
        aug(video,s)
        postfix=postfix+1
    
    coun = coun + 1
    print(f"Videos Augmented : {coun}")

