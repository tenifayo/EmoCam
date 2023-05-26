#Importing necessary libraries
from flask import Flask, render_template, Response, request
import datetime, time
import cv2
from keras.models import load_model
from keras.utils.image_utils import img_to_array
import imutils
from deepface import DeepFace
import numpy as np
from threading import Thread



cap = 0
f1 = 0
f2 = 0
f3 = 0
detect = 0
rec = 0
switch = 1
flip = 0
label = 'hi'

app = Flask(__name__)

emotion_model_path =  r'C:\Users\BAB AL SAFA\Downloads\New folder\emo_model.hdf5'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
emotion_classifier = load_model(emotion_model_path, compile=False)
emotions = ["angry" ,"disgust","fear", "happy", "sad", "surprised", "neutral"]

#Filter functions

def verify_alpha_channel(frame):
    try:
        frame.shape[3]
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    return frame

def apply_filter(frame, r,g,b, intensity=0.5,):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    sepia_bgra = (r,g,b, 1)
    overlay = np.full((frame_h,frame_w, 4), sepia_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.1)
        out.write(rec_frame)


# Start the camera
camera = cv2.VideoCapture(0)

# Define a route to the video feed
def gen_frames():
    global out, cap, rec_frame
    while True:
        ret, frame = camera.read()
        

        if ret:
            if detect:
                frame = imutils.resize(frame,width=300)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
                
                
                frameClone = frame.copy()
                if len(faces) > 0:
                    faces = sorted(faces, reverse=True,
                    key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                    (fX, fY, fW, fH) = faces
                    img = gray[fY:fY + fH, fX:fX + fW]
                    img = cv2.resize(img, (48, 48))
                    img = img.astype("float") / 255.0
                    img_arr = img_to_array(img)
                    img_arr = np.expand_dims(img_arr, axis=0)
                    
                    
                    preds = emotion_classifier.predict(img_arr)[0]
                    global label
                    label = emotions[preds.argmax()]
                    print(label)
                else: 
                    continue

            
                for (i, (emotion, prob)) in enumerate(zip(emotions, preds)):
                            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (255, 255, 255), 1)

                time.sleep(1)


            if flip:
                frame = cv2.flip(frame,1)

            if f1:
                frame = apply_filter(frame, r=112, b=66, g=20)

            if f2:
                frame = apply_filter(frame, r=100, b=100, g=100)

            if f3:
                frame = apply_filter(frame,r=255, b=255, g=0)
            
            if cap:
                now = datetime.datetime.now()
                p = "{}.jpg".format(str(now).replace("-", "").replace(" ","_").replace(":",'').split(".")[0])
                cv2.imwrite(p, frame)
                cap = 0

            if rec:
                rec_frame=frame
                frame= cv2.putText(frame,"", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                



            # if label == "happy":
            #     now = datetime.datetime.now()
            #     p = "{}.jpg".format(str(now).replace("-", "").replace(" ","_").replace(":",'').split(".")[0])

            # Generate camera frames
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except Exception as e:
                pass
        else:
            pass

# Define a route to the index page
@app.route('/')
def index():
    global label
    variable = label
    return render_template('index.html',variable = variable)



# Define a route to the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Define a route to the tasks
@app.route('/tasks',methods=['POST','GET'])
def tasks():
    global switch,camera, label
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global cap
            cap = not cap
        
        elif  request.form.get('f1') == 'filter1':
            global f1
            f1=not f1
        elif  request.form.get('f2') == 'filter2':
            global f2
            f2=not f2
        elif  request.form.get('f3') == 'filter3':
            global f3
            f3=not f3

        elif  request.form.get('flip') == 'flip':
            global flip
            flip = not flip
            
        elif  request.form.get('detect') == 'Detect Emotion':
            global detect
            detect=not detect
            if(detect):
                time.sleep(3) 
        
       
        elif  request.form.get('record') == 'Video':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()

    elif request.method=='GET':
        return render_template('index.html', variable = label)
    return render_template('index.html', variable=label)
                          



if __name__ == '__main__':
    app.run(debug=True)


