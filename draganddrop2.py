import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

image = cv2.imread('/Users/swethankreddy/Downloads/Subject 5.png')
image = cv2.resize(image, (67, 224))

pTime = 0
detector = htm.handDetector()
image_x, image_y = 100, 100  
dragging = False  
offset_x, offset_y = 0, 0

while True:
    success, img = cap.read()
    if not success:
        break
    h, w, c = image.shape
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

   
    if image_y < 0:
        image_y = 0
    if image_x < 0:
        image_x = 0
    if image_y + h > hCam:
        image_y = hCam - h
    if image_x + w > wCam:
        image_x = wCam - w

   
    img[image_y:image_y+h, image_x:image_x+w] = image

    if len(lmList) != 0:
        thumb_x, thumb_y = lmList[4][1], lmList[4][2] 
        index_x, index_y = lmList[8][1], lmList[8][2]  

       
        distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

       
        if (image_x <= thumb_x <= image_x + w and image_y <= thumb_y <= image_y + h) or \
           (image_x <= index_x <= image_x + w and image_y <= index_y <= image_y + h):
            if distance < 50:  
                if not dragging:
                    
                    offset_x = thumb_x - image_x
                    offset_y = thumb_y - image_y
                    dragging = True
               
                target_x = thumb_x - offset_x
                target_y = thumb_y - offset_y
                image_x = int(0.85 * image_x + 0.15 * target_x) 
                image_y = int(0.85 * image_y + 0.15 * target_y)  
            else:
                dragging = False
        else:
            dragging = False

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS : {int(fps)}', (420, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    cv2.waitKey(1)
       

cap.release()
cv2.destroyAllWindows()
