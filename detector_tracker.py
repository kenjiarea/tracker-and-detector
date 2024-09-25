import numpy as np
import cv2

# Initialize the CSRT Tracker
tracker = cv2.TrackerCSRT_create()

# Parameters for YOLO detection and tracking
objectnessThreshold = 0.5  # Objectness threshold
confThreshold = 0.5        # Confidence threshold
nmsThreshold = 0.4         # Non-maximum suppression threshold
inpWidth = 416             # Width of network's input image
inpHeight = 416            # Height of network's input image

# Load class names from file
classesFile = "coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the YOLOv3 network
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Helper function to get the names of output layers from YOLO
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to draw bounding box on frame
def drawPred(left, top, right, bottom, color=(255, 50, 50), thickness=3):
    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

# Function to remove bounding boxes with low confidence and apply NMS
def postprocess(frame, outs):
    frameHeight, frameWidth = frame.shape[:2]
    classIds, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold and classId == 32:  # ClassId 32 is "sports ball"
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    bbox = None
    for i in indices:
        box = boxes[i]  # i is a list
        left, top, width, height = box
        bbox = (left, top, width, height)
        drawPred(left, top, left + width, top + height)
    
    return bbox

# Open video and initialize tracker with the first frame and bounding box
cap = cv2.VideoCapture('soccer-ball.mp4')
ret, frame = cap.read()

if not ret:
    print("Failed to read video")
    cap.release()
    cv2.destroyAllWindows()

red = (0, 0, 255)
blue = (255, 128, 0)

# Get the first detection
blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
net.setInput(blob)
outs = net.forward(getOutputsNames(net))
bbox = postprocess(frame, outs)
ok = tracker.init(frame, bbox)

# Tracking loop
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update tracker
    ret, bbox = tracker.update(frame)
    if ret:
        # Tracking success
        cv2.putText(frame, 'CSRT Tracker', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 2)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, 'Tracking failure detected', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, red, 2)
        count += 1
        
        # Every 5 failures, fallback to YOLO for re-detection
        if count == 5:
            blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            bbox = postprocess(frame, outs)
            
            # Re-initialize tracker with new bbox if detection succeeds
            if bbox is not None:
                tracker.init(frame, bbox)
            count = 0

    # Display frame
    cv2.imshow('image', frame)

    # Break on ESC key press
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
