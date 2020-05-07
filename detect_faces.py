# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
from tensorflow.keras.models import load_model

model  = load_model('models/j.h5')

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt.txt','models/res10_300x300_ssd_iter_140000.caffemodel')

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(input('Enter Image Name: '))
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > 0.5:
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        roi = image[startY:endY,startX:endX]
        roi = cv2.resize(roi,(224,224))
        roi = roi*1./255
        roi = roi.reshape(1,224,224,3)
        result = model.predict(roi).argmax()
        print(result)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        if(result == 1):
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(image,'No Mask', (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        elif(result==0):
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
                cv2.putText(image,'Mask', (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
