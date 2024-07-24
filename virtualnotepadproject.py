import cv2
import numpy as np
import os
import math
import subprocess
import tensorflow as tf
import HandTrackingModule as htm

brushThickness = 50
eraserThickness = 100
MaxBrightness = 100
MinBrightness = 0

model_path = '/Users/swethankreddy/Downloads/swethankmodel.h5'
model = tf.keras.models.load_model(model_path)


def set_volume(volume_level):
    applescript = f'''
    set volume output volume {volume_level}
    '''
    subprocess.call(['osascript', '-e', applescript])


folderPath = "/Users/swethankreddy/Downloads/headerImages"
myList = os.listdir(folderPath)
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]
header = overlayList[0]
drawColor = (0, 0, 255)  


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
volControl = False
predictMode = False

def preprocess_image(image):
    """Preprocess the canvas image for digit prediction."""

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    resized_image = cv2.resize(gray_image, (28, 28))
    
   
    normalized_image = resized_image / 255.0
    
 
    reshaped_image = normalized_image.reshape(1, 28, 28, 1)
    

    cv2.imshow("Resized Image", resized_image)
    
    return reshaped_image



def predict_digit(image):
    """Predict the digit using the pre-trained model"""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        x0, y0 = lmList[4][1:]
        x1, y1 = lmList[8][1:]  
        x2, y2 = lmList[12][1:]
        cx, cy = (x1 + x0) // 2, (y1 + y0) // 2

        fingers = detector.fingersUp()

       
        if volControl:
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x0, y0), 15, (0, 0, 0), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 0, 0), cv2.FILLED)
                cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0), 3)
                cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                length = math.hypot(x1 - x0, y1 - y0)
                vol = np.interp(length, [50, 250], [MinBrightness, MaxBrightness])
                volBar = np.interp(length, [50, 250], [600, 250])
                volPer = np.interp(length, [50, 250], [0, 100])
                set_volume(int(volPer)) 
                color = (0, 0, 255) 
                
                if length < 50:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                cv2.rectangle(img, (50, 250), (85, 600), color, 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 600), color, cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (45, 220), cv2.FONT_HERSHEY_COMPLEX, 1, color, 3)

       
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            if y1 < 125:
                if 690 < x1 < 730:
                    volControl = False
                    header = overlayList[6]
                    drawColor = (0, 0, 255)
                elif 790 < x1 < 830:
                    volControl = False
                    header = overlayList[7]
                    drawColor = (220, 35, 35)
                elif 890 < x1 < 930:
                    volControl = False
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 990 < x1 < 1030:
                    volControl = False
                    header = overlayList[3]
                    drawColor = (255, 255, 255)
                elif 1090 < x1 < 1165:
                    volControl = False
                    header = overlayList[5]
                    drawColor = (0, 0, 0)
                elif 150 < x1 < 250:
                    header = overlayList[4]
                    drawColor = False
                    volControl = True
                elif 330 < x1 < 460:
                    header = overlayList[1]
                    drawColor = False
                    volControl = False
                    if not predictMode:
                        predictMode = True
                        print("Predict mode activated")
                        
                        digit = predict_digit(imgCanvas)
                        print(f'Predicted Digit: {digit}')
                        cv2.putText(img, f'Predicted Digit: {digit}', (50, 650), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                        cv2.imshow("Image", img)
                        cv2.waitKey(2000)
                        predictMode = False

       
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

   
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    
    key = cv2.waitKey(1) & 0xFF  # Capture the key press

    if key == ord('s'):
        cv2.imwrite('my_painting.png', imgCanvas)
        print("Canvas saved as my_painting.png")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
