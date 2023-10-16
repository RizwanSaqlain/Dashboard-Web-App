from flask import Flask, Request, render_template, Response, jsonify, session
from ultralytics import YOLO
import requests
from icecream import ic
from urllib.parse import quote
import cv2
import time
import multiprocessing


app = Flask(__name__)
model=YOLO("yolov8_100epochs.pt")
camera = None
camera_active = False
frame_queue = multiprocessing.Queue(maxsize=30)
frame_byte_queue = multiprocessing.Queue(maxsize=30)

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

def check_camera():
    global camera_active
    if camera_active == True:
        camera.release()
        camera_active = False

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



def instant_generate_frames():
    frame_count = 0
    start_time=time.time()
    while True:
        success, frame = camera.read() 
        if not success:   
            break

        else:
            frame_count += 1
            detected_objects = custom_yolo_detection(frame)
            annotated_frame, class_count = annotate_frame(frame, detected_objects)

                
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}    Total Persons:{class_count[5]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            ret, buffer = cv2.imencode('.jpg',annotated_frame)
            if not ret:
                break

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        

        #time.sleep(0.008)  # Adjust the sleep time to control the frame rate'''



def capture_frames(q):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        q.put(frame)

    cap.release()

def generate_frames(q):
    global frame_byte_queue
    frame_count = 0
    thickness = 2
    start_time=time.time()
    while True:
        if not q.empty():
            frame = q.get()
            frame_count += 1
            detected_objects = model.predict(source=frame, conf=0.3, verbose=False)
            for result in detected_objects:
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


            #annotated_frame, class_count = annotate_frame(frame, detected_objects)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 1 / elapsed_time
            #print(f"Frame Rate: {fps:.2f} FPS")
            cv2.putText(frame, f"FPS: {fps:.2f}    Total Persons:{class_count[5]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            ret, buffer = cv2.imencode('.jpg',frame)
            if not ret:
                break

            frame_bytes = buffer.tobytes()
            frame_byte_queue.put(frame_bytes)
            return

        
        

        #time.sleep(0.008)  # Adjust the sleep time to control the frame rate


def send_frames():
    global frame_queue
    global frame_byte_queue
    while True:
        generate_frames(frame_queue)
        if not frame_byte_queue.empty():
            frame_bytes = frame_byte_queue.get()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            


'''@app.route("/login", methods=['GET', 'POST'])
def login():
    if Request.method == 'POST':
        pass

    return render_template("login.html")'''

@app.route("/")
@app.route("/dashboard", methods=['GET','POST'])
def dashboard():
    check_camera()
    return render_template("dashboard.html")

@app.route("/tables", methods=['GET','POST'])
def tables():
    check_camera()
    return render_template("tables.html")


@app.route("/news", methods=['GET','POST'])
def news():
    check_camera()
    return render_template("news.html")

@app.route("/employees", methods=['GET','POST'])
def employees():
    check_camera()
    return render_template("employees.html")

@app.route("/CCTV", methods=['GET','POST'])
def CCTV():
    global camera
    global camera_active
    if camera_active == False:
        camera = cv2.VideoCapture(0)
        while True:
            if camera.isOpened() == True:
                break
            time.sleep(1)
        time.sleep(2)
        camera_active = True
    return render_template("CCTV.html") 

@app.route("/weatherReports", methods=['GET','POST'])
def weatherReports():
    check_camera()
    base_url = 'http://api.weatherapi.com/v1'
    api_method = '/current.json'
    city='New Delhi'
    api_key = 'b4de7d54fab44efbb39204156231310'
    city = quote(city)
    url= f'{base_url}{api_method}?key={api_key}&q={city}'

    response = requests.get(url)
    data = response.json()
    ic(response)
    ic(response.json())
    return render_template("weatherReports.html",weather_data=data)

@app.route('/video_feed')
def video_feed():
    capture_process = multiprocessing.Process(target=capture_frames, args=(frame_queue,))
    capture_process.start()
    return Response(send_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')       #When using instant_generate_frame remove capture process
    '''return Response(instant_generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')'''

if __name__ == '__main__':
    app.run(debug=True)