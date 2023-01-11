import cv2
from keras.models import load_model
import numpy as np

blue = (255, 0, 0)
green = (0, 255, 0)

#video_capture = cv2.VideoCapture("test.mp4")
video_capture = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model("model")


while True:

    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_rects = face_detector.detectMultiScale(gray, 1.1, 8)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), blue, 2)
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (32, 32))
        roi = roi / 255.0

        roi = roi[np.newaxis, ...]
        prediction = model.predict(roi)
        label = "Sonriendo" if prediction >= 0.5 else "No Sonrisa"
        
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.75, green, 2)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()