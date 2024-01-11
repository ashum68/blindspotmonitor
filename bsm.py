from ultralytics import YOLO
import cv2, math, numpy as np, cvzone

# cap = cv2.VideoCapture(0) # for webcam
cap = cv2.VideoCapture("./test.mov") # for video file

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (360, 640))

YOLOmodel = YOLO('yolov8n.pt')

vehicle_type = ['person', 'bicycle', 'car', 'motorbike', 'n/a', 'bus', 'n/a', 'truck']

detect_zone = np.array([[50,320],[250,320],[350,370],[350,620],[170,620]], np.int32)
detect_zone = detect_zone.reshape((-1,1,2))

warning = cv2.imread('bsm.png')
img_height, img_width, _ = warning.shape

while cap.isOpened():
    success, img = cap.read()
    results = YOLOmodel(img, stream=True)
    
    # draw detection zone
    cv2.polylines(img,[detect_zone],True,(0,255,255),thickness=4)
    cv2.circle(img, (280,355), 3, (0,255,0), cv2.FILLED)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # get object that was detected
            obj_detected = int(box.cls[0])

            # confidence
            conf = math.ceil((box.conf[0]*100))/100

            # check if object is a valid vehicle
            if obj_detected == 1 or obj_detected == 2 or obj_detected == 3\
                or obj_detected == 5 or obj_detected == 7 and conf>0.4:

                # get position of object
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                cx = x1 + (x2-x1)//2

                # approaching car marker
                cv2.line(img, (x1,y2), (x2,y2), (255,0,0), 5)
                cv2.circle(img, (cx, y2), 3, (0,255,0), cv2.FILLED)

                # check if in blind spot zone
                if 50 < cx < 350 and 320 < y2 < 620:

                    # display warning
                    img[ 20:20+img_height , 130:130+img_width ] = warning
                    cvzone.putTextRect(img, f'Watch for {vehicle_type[obj_detected]}!', (125, 125), scale=0.9, thickness=1, offset=0)
                
    
    # display, press q to quit
    if success == True:
        out.write(img)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out.release()





