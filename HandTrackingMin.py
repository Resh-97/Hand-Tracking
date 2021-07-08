import cv2
import mediapipe as mp
import time
cam = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
## static_image_mode if False will track only when the confidence falls.
## True is for continuour tracking but this slows down the processing.
hands = mpHands.Hands()  # Using default values of the Hands class
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cam.read()
    imgRGB  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Original Images are BGR
    results = hands.process(imgRGB)  #Processes RGB and returns two fields: "multi_hand_landmarks" & "multi_handedness"
    #print(results.multi_hand_landmarks)

    ##Checking if there are multiple hands and extracting each of them
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark): ##lms is the ratio of height and width coord.
                #print(id,lm)
                height, width, channel = img.shape
                x_center, y_center = int(lm.x * width), int(lm.y * height)
                print(id, x_center, y_center)
                if id ==4:
                    cv2.circle(img, (x_center, y_center), 15,(255,0,255), cv2.FILLED )
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #Draws the keypoints

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
