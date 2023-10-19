from flask import Flask, Request, render_template, Response, jsonify, session
from ObjectDetection import instant_generate_frames, capture_frames, send_frames
from API_calls import get_currentWeatherReports
import cv2
import time
import multiprocessing


app = Flask(__name__)
camera = None
camera_active = False
frame_queue = multiprocessing.Queue(maxsize=5)
frame_byte_queue = multiprocessing.Queue(maxsize=5)


def check_camera():
    global camera_active
    if camera_active == True:
        camera.release()
        camera_active = False
            

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
    data = get_currentWeatherReports('New Delhi')
    return render_template("weatherReports.html",weather_data=data)

@app.route('/video_feed')
def video_feed():
    #capture_process = multiprocessing.Process(target=capture_frames, args=(frame_queue,))
    #capture_process.start()
    #return Response(send_frames(frame_queue,frame_byte_queue), mimetype='multipart/x-mixed-replace; boundary=frame')       #When using instant_generate_frames() remove capture process
    return Response(instant_generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)