import cv2
import math
import numpy as np
import HandTrackingModule as htm
import subprocess

def set_volume(volume_level):
    applescript = f'''
    set volume output volume {volume_level}
    '''
    subprocess.call(['osascript', '-e', applescript])


cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.resize(img, (1300, 700))
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    dis = 0
    if len(lmList) > 0:
        
        cv2.line(img, (lmList[4][1], lmList[4][2]), (lmList[8][1], lmList[8][2]), (255, 0, 0), 2)
        cv2.circle(img, (lmList[4][1], lmList[4][2]), 10, (0, 0, 255), 10)
        cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (0, 0, 255), 10)
        cv2.circle(img, (int((lmList[4][1] + lmList[8][1]) / 2), int((lmList[4][2] + lmList[8][2]) / 2)), 10, (100, 100, 100), 5)
        
       
        dis = math.hypot(lmList[4][1] - lmList[8][1], lmList[4][2] - lmList[8][2])

   
    cv2.rectangle(img, (75, 75), (125, 525), (255, 0, 0), 2)
    
    
    sound = int((2 * dis - 100) / 5)
    sound = max(0, min(100, sound))
    
    
    x, y = 100, int(500 - (4 * sound))
    
   
    if sound < 20:
        color = (0, 255, 0)
    elif sound > 80:
        color = (0, 0, 255)
    else:
        color = (0, 255, 255)

   
    set_volume(sound)

   
    cv2.rectangle(img, (x - 25, y - 25), (x + 25, y + 25), color, -1)
    cv2.putText(img, f"Sound: {sound}%", (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    
    cv2.imshow("Image", img)

   
    cv2.waitKey(1)
   

cap.release()
cv2.destroyAllWindows()
