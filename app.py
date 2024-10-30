import cv2
import numpy as np
from flask import Flask, Response, render_template
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

glasses_images = [cv2.imread(f'filters/glasses{i}.png', -1) for i in range(1, 7)]
glasses_index = 0
count = 0

next_button_x, next_button_y, button_width, button_height = 50, 50, 100, 50
capture_button_x, capture_button_y = 500, 100


def draw_buttons(frame):
    cv2.rectangle(frame, (next_button_x, next_button_y), (next_button_x + button_width, next_button_y + button_height),
                  (0, 255, 0), -1)
    cv2.putText(frame, "Next", (next_button_x + 15, next_button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                2)


    cv2.rectangle(frame, (capture_button_x, capture_button_y),
                  (capture_button_x + button_width, capture_button_y + button_height), (0, 0, 255), -1)
    cv2.putText(frame, "Capture", (capture_button_x + 10, capture_button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)


def detect_hand_near_button_next(hand_landmarks, button_x, button_y):
    print(f'buttonx : {button_x} and buttony : {button_y}')
    if hand_landmarks:
        finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = int(finger_tip.x * frame_width), int(finger_tip.y * frame_height)
        print(f'x:{x} and y:{y}')
        if 50 < x < 150 and 100 < y < 200:
            print('button detected')
            return True
    return False

def detect_hand_near_button_capture(hand_landmarks, button_x, button_y):
    print(f'buttonx : {button_x} and buttony : {button_y}')
    if hand_landmarks:
        finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = int(finger_tip.x * frame_width), int(finger_tip.y * frame_height)
        print(f'x:{x} and y:{y}')
        if 300 < x < 600 and 100 < y < 200:
            print('captured')
            return True
    return False

def overlay_glasses(frame, x, y, w, h):
    global glasses_index
    glasses = glasses_images[glasses_index]

    glasses_resized = cv2.resize(glasses, (w, int(glasses.shape[0] * w / glasses.shape[1])))
    overlay_y = y + int(h * 0.2)

    mask = glasses_resized[:, :, 3]
    mask_inv = cv2.bitwise_not(mask)

    roi = frame[overlay_y:overlay_y + glasses_resized.shape[0], x:x + glasses_resized.shape[1]]

    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    filter_fg = cv2.bitwise_and(glasses_resized[:, :, :3], glasses_resized[:, :, :3], mask=mask)

    frame[overlay_y:overlay_y + glasses_resized.shape[0], x:x + glasses_resized.shape[1]] = roi_bg + filter_fg


def generate_frames():
    video = cv2.VideoCapture(0)
    global glasses_index, frame_width, frame_height, count

    while True:
        success, frame = video.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]

        overlay_glasses(frame, x, y, w, h)
        frame_height, frame_width, _ = frame.shape

        draw_buttons(frame)

        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(finger_tip.x * frame_width), int(finger_tip.y * frame_height)

                #print(f"Index Finger Tip coordinates: x={x}, y={y}")

                if detect_hand_near_button_next(hand_landmarks, next_button_x, next_button_y):
                    glasses_index = (glasses_index + 1) % len(glasses_images)

                if detect_hand_near_button_capture(hand_landmarks, capture_button_x, capture_button_y):
                    print("Capture button detected!")

                    count += 1
                    image_name = f'image{count}.jpg'
                    cv2.imwrite(image_name, frame)
                    print(f"Image captured: {image_name}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
