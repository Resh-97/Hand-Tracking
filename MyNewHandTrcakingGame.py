import cv2
import mediapipe as mp
import time
from HandTrackingModule import handDetector
cam = cv2.VideoCapture(0)



prevTime = 0
currTime = 0
detector = handDetector()
while True:
    success, img = cam.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0 :
        print(lmList[4]) # Prints the specific landmark position of the defined hand.

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
