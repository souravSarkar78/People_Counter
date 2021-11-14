import time
import cv2 
from flask import Flask, render_template, Response
from utils import *


app = Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html')


def video():
    """Video processing function"""
    cap = cv2.VideoCapture('video99.mp4')

    # Reset count and object list
    reset()
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            img = objectTracker(img)

            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.02)
        else: 
            break
        

@app.route('/video_feed')
def video_feed():

    """Processed video that binds to the image src element"""
    return Response(video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False)