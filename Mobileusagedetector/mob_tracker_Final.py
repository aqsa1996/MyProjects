import cv2
import csv
import time 
import Centroid as cd 
from Centroid import CentroidTracker
import argparse
import numpy as np
from datetime import datetime
from rect import rectdistance
#creating table
import mysql.connector
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Hello@123", #change passwd 
  database="vbasedsys"
)

mycursor = mydb.cursor()



ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', 
                help = 'path to yolo config file', default='/path/to/yolov3-tiny.cfg')
ap.add_argument('-w', '--weights', 
                help = 'path to yolo pre-trained weights', default='/path/to/yolov3-tiny_10000.weights')
ap.add_argument('-cl', '--classes', 
                help = 'path to text file containing class names',default='/path/to/objects.names')
args = ap.parse_args()

ct = CentroidTracker()
# Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

frame_no=0
# Darw a rectangle surrounding the object and its class name 
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
# Define a window to show the cam stream on it
window_title= "mobile Detector"   
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)


# Load names classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


#Generate color for each class randomly
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Define network from configuration file and load the weights from the given weights file
net = cv2.dnn.readNet(args.weights,args.config)

# Define video capture for default cam
cap = cv2.VideoCapture('vid.mp4')

count=0
while cv2.waitKey(1) < 0:
    
    hasframe, image = cap.read()
    #image=cv2.resize(image, (620, 480)) 
    frame_no+=1
    print('frame:',frame_no)
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
    Width = image.shape[1]
    Height = image.shape[0]
    net.setInput(blob)
    
    outs = net.forward(getOutputsNames(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    
    #print(len(outs))
    
    # In case of tiny YOLOv3 we have 2 output(outs) from 2 different scales [3 bounding box per each scale]
    # For normal normal YOLOv3 we have 3 output(outs) from 3 different scales [3 bounding box per each scale]
    
    # For tiny YOLOv3, the first output will be 507x6 = 13x13x18
    # 18=3*(4+1+1) 4 boundingbox offsets, 1 objectness prediction, and 1 class score.
    # and the second output will be = 2028x6=26x26x18 (18=3*6) 
    
    for out in outs: 
        #print(out.shape)
        for detection in out:
            
        #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
            scores = detection[5:]#classes scores starts from index 5
            class_id = np.argmax(scores)
          
            confidence = scores[class_id]
           
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # apply  non-maximum suppression algorithm on the bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    p_box=[]
    m_box=[]
    rects=[]
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        box1=(x,y,x+w,y+h)
       
       
        if str(classes[class_ids[i]])=='person':
         p_box.append(box)
         rects.append(box1)

         
        if str(classes[class_ids[i]])=='cell phone':
         m_box.append(box)
    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
     text = "ID {}".format(objectID)
     #print(text)
     for i in p_box:
      for j in m_box:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        x1= j[0]
        y1= j[1]
        w1= j[2]
        h1= j[3]
        try:
         dist=rectdistance(x, y, x+w, y+h, x1, y1, x1+w1, y1+h1)
        except:
         dist=100
        if (dist==0):
         now = datetime. now()
         print(now,'person', text,'is using cellphone')
         
         sql = "INSERT INTO mobileusage (personID, FrameNo) VALUES (%s, %s)"
        
         val = (str(objectID), str(frame_no))
         mycursor.execute(sql, val)

         mydb.commit()

         print(mycursor.rowcount, "record inserted.")
         
         #count+=1
        #else:
         #count=0
        
     #if (count==25):
      #  print('person is using cellphone continuously')
        

      
        
   
    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
    cv2.imshow(window_title, image)

