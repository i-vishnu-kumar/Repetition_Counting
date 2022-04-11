import numpy as np
import cv2

a = np.load("3D CNN\\Data\\Stitch\\Left\\48.npy")
print(a)
for i in a :
    i = cv2.resize(i,(1024,1024)) 
    cv2.imshow("",i)
    cv2.waitKey(0) 
cv2.destroyAllWindows
