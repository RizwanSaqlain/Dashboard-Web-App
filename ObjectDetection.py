from ultralytics import YOLO
import numpy as np
import time
import cv2

class_labels = {
    0: ['Hardhat', (57, 47, 140)],
    1: ['Mask',(231,236,239)],
    2: ['NO-Hardhat',(57, 47, 140)],
    3: ['NO-Mask',(231,236,239)],
    4: ['NO-Safety Vest',(0,0,255)],
    5: ['Person',(0,230,0)],
    6: ['Safety Cone',(116,191,149)],
    7: ['Safety Vest',(0,0,255)],
    8: ['machinery',(32,255,255)],
    9: ['vehicle',(32,255,255)]
}


model = YOLO("yolov8_100epochs.pt")

def custom_yolo_detection(frame):
    results=model.predict(source=frame, conf=0.3, verbose=False)
    return results


def annotate_frame(frame, detection_results):  
    global class_count
    thickness = 2
    
    for result in detection_results:
        #class_labels=result.names                                             #To use when class names unknown, adds extra computation
        classes_present=result.boxes.cls.tolist()
        bbox = result.boxes.xyxy.tolist()  
        confidences=result.boxes.conf.tolist()
        class_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for (class_id, xyxy, conf) in zip(classes_present, bbox, confidences):
            class_count[class_id] += 1
            class_name=class_labels[class_id][0]                                 #Detected class names
            x1, y1, x2, y2 = [int(xyxy[i]) for i in range(4)] 
            confidence = str(round(conf,2))
            color=class_labels[class_id][1]


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 10                                                       #Label location adjustment
            cv2.putText(frame, class_name + ' ' + confidence, (text_x, text_y), font, font_scale, color, font_thickness)

    return frame, class_count



def instant_generate_frames(camera):
    start_time=time.time()
    while True:
        success, frame = camera.read() 
        if not success:   
            break

        else:
            detected_objects = custom_yolo_detection(frame)
            annotated_frame, class_count = annotate_frame(frame, detected_objects)

                
            end_time = time.time()
            fps = 1 / np.round(end_time - start_time ,2)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}       Total Persons:{class_count[5]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            ret, buffer = cv2.imencode('.jpg',annotated_frame)
            if not ret:
                break

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        

        #time.sleep(0.008)  # Adjust the sleep time to control the frame rate'''



def capture_frames(frame_queue):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

    cap.release()

def generate_frames(frame_queue,frame_byte_queue):
    thickness = 2
    start_time=time.time()
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            detected_objects = model.predict(source=frame, conf=0.3, verbose=False)

            for result in detected_objects:
                #class_labels=result.names                                             #To use when class names unknown, adds extra computation
                classes_present = result.boxes.cls.tolist()
                bbox = result.boxes.xyxy.tolist()  
                confidences = result.boxes.conf.tolist()
                class_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

                for (class_id, xyxy, conf) in zip(classes_present, bbox, confidences):
                    class_count[class_id] += 1
                    class_name=class_labels[class_id][0]                                 #Detected class names
                    x1, y1, x2, y2 = [int(xyxy[i]) for i in range(4)] 
                    confidence = str(np.round(conf,2))
                    color=class_labels[class_id][1]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
                    text_x = x1
                    text_y = y1 - 10                                                       #Label location adjustment
                    cv2.putText(frame, class_name + ' ' + confidence, (text_x, text_y), font, font_scale, color, font_thickness)
            
            end_time = time.time()
            fps = 1 / np.round(end_time - start_time ,2)
            #print(f"Frame Rate: {fps:.2f} FPS")
            cv2.putText(frame, f"FPS: {fps:.2f}      Total Persons:{class_count[5]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            ret, buffer = cv2.imencode('.jpg',frame)
            if not ret:
                break

            frame_bytes = buffer.tobytes()
            frame_byte_queue.put(frame_bytes)
            return

        #time.sleep(0.008)  # Adjust the sleep time to control the frame rate


def send_frames(frame_queue, frame_byte_queue):
    while True:
        generate_frames(frame_queue, frame_byte_queue)
        if not frame_byte_queue.empty():
            frame_bytes = frame_byte_queue.get()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')