import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from PIL import Image
import csv


#stage6

def getpred(red,yello,green):
  # red = cv2.imread('content/Red.jpg')
  # yello = cv2.imread('content/orange.jpg')
  # green = cv2.imread('content/Green.jpg')
  countR = np.sum(red==0)
  countY = np.sum(yello==0)
  countG = np.sum(green==0)
  
  if (countR < countG and countR < countY):
    predict = 'Stop'
  if (countG < countR and countG < countY):
    predict = 'GO'
  if (countY < countG and countY < countR):
    predict = 'Ready'
  print('RED-Black:', countR, '\n', 'Orange', countY, '\n','Green', countG, '\n', predict)
  return predict



#stage5

def getDATA():
  im = cv2.imread('content/ROI.jpg')
  cv2.imshow('RGB', im)
  image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  imagecopy = image
  imagecopy2 = image

  lower = np.array([170, 0, 0])
  upper = np.array([255, 255, 255])
  mask = cv2.inRange(image, lower, upper)
  result = cv2.bitwise_and(image, image, mask=mask)
  cv2.imwrite('content/Red.jpg', result)

  lowerY = np.array([10, 100, 20])
  upperY = np.array([25, 255, 255])
  maskY = cv2.inRange(imagecopy2, lowerY, upperY)
  result1 = cv2.bitwise_and(imagecopy2, imagecopy2, mask=maskY)
  cv2.imwrite('content/orange.jpg', result1)

  lowerG = np.array([36, 100, 25])
  upperG = np.array([85, 255, 255])
  maskG = cv2.inRange(imagecopy, lowerG, upperG)
  result2 = cv2.bitwise_and(imagecopy, imagecopy, mask=maskG)
  cv2.imwrite('content/Green.jpg', result2)


  cv2.imwrite("content/out.png", im)
  cv2.imshow("MASK RED", result)
  cv2.imshow("MASK Yello", result1)
  cv2.imshow("MASK GREEN", result2)

  prediction = getpred(result, result1, result2)

  return prediction


#stage4

def dp(frame, classId, conf, left, top, right, bottom, classes,str_prediction):
  cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 3)
  cv2.putText(frame, str_prediction, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
  #print(left, top, right, bottom)
  #getROI(frmecopy, left,top,right,bottom)

  return frame

#stage3

def BBox(frame_org, outs, confThreshold, classes, nmsThreshold,):
  frameHeight = frame_org.shape[0]
  frameWidth = frame_org.shape[1]
  classIds = []
  confidences = []
  boxes = []
  for out in outs:
    for detection in out:
      scores = detection[5:]
      classId = np.argmax(scores)
      confidence = scores[classId]
      # print(classes[classId])
      # print(str(confidence))
      if confidence > confThreshold and classes[classId] == "traffic light":
          # if classes[classId] == "person":
          center_x = int(detection[0] * frameWidth)
          center_y = int(detection[1] * frameHeight)
          width = int(detection[2] * frameWidth)
          height = int(detection[3] * frameHeight)
          left = int(center_x - width / 2)
          top = int(center_y - height / 2)
          classIds.append(classId)
          confidences.append(float(confidence))
          boxes.append([left, top, width, height])
  indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
  for i in indices:
      i = i
      box = boxes[i]
      left = box[0]
      top = box[1]
      width = box[2]
      height = box[3]
      if_predict = True
      if if_predict:
          cv2.imwrite('content/Frame.jpg', frame_org)
          img = cv2.imread('content/Frame.jpg')
          ROI = img[top:top + height, left:left + width]
          cv2.imwrite('content/ROI.jpg', ROI)
          str_prediction = getDATA()
          frame_org = dp(frame_org, classIds[i], confidences[i], left, top, left + width, top + height, classes, str_prediction)  

  return frame_org
 



#stage 2
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

#Stage 1
def getTL(frame):
   confThreshold = 0.5  
   nmsThreshold = 0.4  
   inpWidth = 416  
   inpHeight = 416  
   # Load names of classes
   classesFile = r"COCO.names"
   classes = None
   with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

   modelConfiguration = r"yolov3.cfg"
   modelWeights = r"yolov3.weights"

   net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
   blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
   net.setInput(blob)
   outs = net.forward(getOutputsNames(net))
   print("Processed TF")
   img = BBox(frame, outs, confThreshold, classes, nmsThreshold)
   return img

#main stage

if __name__ == '__main__':
  cap = cv2.VideoCapture('TFL.mp4')
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  size = (frame_width, frame_height)
  count = 0
  fps = 0

  while True:
      hasFrame, frame = cap.read()
      if not hasFrame:
        break
      if fps!=30:
        fps+=1      
      else:
        imS = cv2.resize(frame, (960, 540))
        imo = getTL(imS)
        #cv2.imshow('frame', imo)
        cv2.waitKey(100)
        print('frame', count)
        count = count+1
        fps=0