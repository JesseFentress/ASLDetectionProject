import cv2
import threading
import copy
import itertools
import numpy as np
from hello.mediapipe_hands import MediaPipeHands
from hello.model import Model

class VideoCamera(object):

    def __init__(self):
        self.mp = MediaPipeHands()
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        with self.mp.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as hands:
            while True:
                (self.grabbed, self.frame) = self.video.read()
                (x, y, c) = self.frame.shape
                if not self.grabbed:
                    continue
                self.frame.flags.writeable = False
                results = hands.process(self.frame)
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.frame.flags.writeable = True
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                if results.multi_hand_landmarks:
                    landmarks = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            lmx = np.float32(lm.x * x)
                            lmy = np.float32(lm.y * y)
                            landmarks.append([lmx, lmy])
                        self.mp.mp_drawing.draw_landmarks(
                            self.frame,
                            hand_landmarks,
                            self.mp.mp_hands.HAND_CONNECTIONS,
                            self.mp.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp.mp_drawing_styles.get_default_hand_connections_style())
                        model = Model()
                        
def generate_video(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def preprocess_hand(landmark):
    l = copy.deepcopy(landmark)
    base_x, base_y = 0, 0
    for index, point in enumerate(l):
        if index == 0:
            base_x, base_y = point[0], point[1]
            l[index][0] = l[index][0] - base_x
            l[index][1] = l[index][1] - base_y
    l = list(itertools.chain.from_iterable(l))
    max_value = max(list(abs, l))
    def normalize(n):
        return n / max_value
    l = list(map(normalize, l))
    return l




