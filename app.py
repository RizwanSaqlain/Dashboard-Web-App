from flask import Flask, Request, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import time


app = Flask(__name__)
model=YOLO("yolov8_100epochs.pt")
camera = None
camera_active = False


class_labels = {
    0: 'Hardhat',
    1: 'Mask',
    2: 'NO-Hardhat',
    3: 'NO-Mask',
    4: 'NO-Safety Vest',
    5: 'Person',
    6: 'Safety Cone',
    7: 'Safety Vest',
    8: 'machinery',
    9: 'vehicle'
}
label_colors={
    0: (57, 47, 140),
    1: (231,236,239),
    2: (57, 47, 140),
    3: (231,236,239),
    4: (0,0,255),
    5: (0,230,0),
    6: (116,191,149),
    7: (0,0,255),
    8: (32,18,70),
    9: (32,18,70)
}

def check_camera():
    global camera_active
    if camera_active == True:
        camera.release()
        camera_active = False

def custom_yolo_detection(frame):
    results=model.predict(source=frame, conf=0.3)
    return results


def annotate_frame(frame, detection_results):  
    thickness = 2
    
    for result in detection_results:
        #class_labels=result.names                                             #To use when class names unknown, adds extra computation
        classes_present=result.boxes.cls.tolist()
        bbox = result.boxes.xyxy.tolist()  
        confidences=result.boxes.conf.tolist()
        for (class_id, xyxy, conf) in zip(classes_present, bbox, confidences):
            class_name=class_labels[class_id]                                  #Detected class names
            x1, y1, x2, y2 = [int(xyxy[i]) for i in range(4)] 
            confidence = str(round(conf,2))
            color=label_colors[class_id]


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 10                                                       #Label location adjustment
            cv2.putText(frame, class_name + ' ' + confidence, (text_x, text_y), font, font_scale, color, font_thickness)

        return frame



def generate_frames():
    count=0
    frame_count = 0
    start_time=time.time()
    while True:
        success, frame = camera.read() 
        if not success:   
            break

        else:
            frame_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            detected_objects = custom_yolo_detection(frame)
            annotated_frame = annotate_frame(frame, detected_objects)

            '''if frame_count % 2 == 0 :
                detected_objects = custom_yolo_detection(frame)
                annotated_frame = annotate_frame(frame, detected_objects)
            else:
                annotated_frame=frame'''
                
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            ret, buffer = cv2.imencode('.jpg',annotated_frame)
            if not ret:
                break

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        #time.sleep(0.008)  # Adjust the sleep time to control the frame rate




@app.route("/")
def index():
    check_camera()
    return render_template("dashboard.html")

'''@app.route("/login", methods=['GET', 'POST'])
def login():
    if Request.method == 'POST':
        pass

    return render_template("login.html")'''

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
        time.sleep(2)
        camera_active = True
    
    return render_template("CCTV.html") 

@app.route("/weatherReports", methods=['GET','POST'])
def weatherReports():
    check_camera()
    return render_template("weatherReports.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)