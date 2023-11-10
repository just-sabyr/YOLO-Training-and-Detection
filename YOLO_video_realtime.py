### Real-Time object detection using YOLOv8

from ultralytics import YOLO
import cv2
import math
import cvzone # for putting text on frames

cap = cv2.VideoCapture(0) # camera ID
model = YOLO('runs/detect/train2/weights/best.pt')

while True:
    # Capturing video frame by frame
    ret, frame = cap.read()   

    results = model(frame)
   
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Inference', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

"""

# Does almost the same as the code above, only the labels can be reduced using classes.txt file


# Following will work only if classes.txt is ONLY top-cut from original labels in model (coco) github

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().strip().splitlines()

while True:
    # Capturing video frame by frame
    ret, frame = cap.read()   

    results = model(frame, stream=True)

    
    for info in result:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            names = box.cls[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil(conf * 100)
            names = int(names)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            try:
                cvzone.putTextRect(frame, f'{classnames[names]} {conf}%', [x1+8, y1-12], scale=1.5, thickness=2) 
            except IndexError:
                cvzone.putTextRect(frame, 'out of classnames', [x1+8, y1-12], scale=1.5, thickness=2)

    cv2.imshow('YOLOv8 Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""