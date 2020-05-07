# importing libraries 
import numpy as np
import argparse
import cv2
from tensorflow.keras.models import load_model
import imutils
model  = load_model('models/j.h5')

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt.txt','models/res10_300x300_ssd_iter_140000.caffemodel')


# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('short.mp4') 

# Check if camera opened successfully 
if (cap.isOpened()== False):
    print("Error opening video file") 

# Read until video is completed 
while(cap.isOpened()):
    ret, frame = cap.read() 
    if ret == True:
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        #cv2.imshow('Frame',frame)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
 
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            roi = frame[startY:endY,startX:endX]
            roi = cv2.resize(roi,(224,224))
            roi = roi * 1./255
            roi = roi.reshape(1,224,224,3)
            result = model.predict(roi).argmax()
            y = startY - 10 if startY - 10 > 10 else startY + 10
            if(result == 1):
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(frame,'No Mask', (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            elif(result==0):
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
                cv2.putText(frame,'Mask', (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
 
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else:
        break

# When everything done, release 
# the video capture object 
cap.release() 

# Closes all the frames 
cv2.destroyAllWindows() 
