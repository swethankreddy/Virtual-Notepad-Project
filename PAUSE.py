import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector()
tipIds = [4, 8, 12, 16, 20]
pTime = 0
pause = False
overLayList = []

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False) 
    
    if len(lmList) != 0:
        fingers = [lmList[8][2] < lmList[6][2], lmList[12][2] < lmList[10][2], lmList[16][2] < lmList[14][2], lmList[20][2] < lmList[18][2], lmList[4][1] > lmList[3][1]]
        totalFingers = fingers.count(True)
        print(totalFingers)
        
        if totalFingers > 3:
            pause = True
        elif totalFingers <= 3:
            pause = False

    if not pause:
        if len(overLayList)!=0:
         h, w, c = overLayList[0].shape
         img[0:h, 0:w] = overLayList[0]
       
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        
        cv2.putText(img, f'FPS : {int(fps)}', (470, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
    
        cv2.waitKey(1)
        

cap.release()
cv2.destroyAllWindows()
     