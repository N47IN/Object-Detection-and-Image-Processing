import cv2
import argparse
import numpy as np
import imutils
import time
ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

webcam = cv2.VideoCapture(0)

writer = None
(W, H) = (None, None)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(args.weights, args.config)
# try to determine the total number of frames in the video file
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h,center_point):

    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# run inference through the network
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()
    (Width,Height)=frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    center_point = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
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
                center_point.append([(center_x, center_y),class_id])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detections = 0
# go through the detections remaining
# after nms and draw bounding box
    for i in indices:
        i = i
    
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
    
        draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),center_point[detections])
        detections+=1
    cv2.imshow("object detection", frame)

# wait until any key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


   

    
