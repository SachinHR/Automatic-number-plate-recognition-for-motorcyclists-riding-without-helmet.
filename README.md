# Automatic-number-plate-recognition-for-motorcyclists-riding-without-helmet.

## Description
Real-time Number plate recognition for motorcyclists riding without helmet project with OpenCV and Python

* Recognising Motor Cyclist Riding Without Helmet
```
import cv2

motor_cycle = cv2.CascadeClassifier('haarcascade_motorcyclist_without_helmet.xml')                     
number_plate = cv2.CascadeClassifier('haarcascade_number_plate.xml')

cap = cv2.VideoCapture(0)                                                     

while 1: 
    ret, img = cap.read()                                               
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    motor_cycle = motor_cycle.detectMultiScale(gray, 1.3, 5)                                           

    for (x,y,w,h) in motor_cycle:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)         
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Motor_Cyclist_Without_Helmet',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
```

![Image](https://github.com/SachinHR/Automatic-number-plate-recognition-for-motorcyclists-riding-without-helmet./blob/master/Image/Motor_Cycle.png) 

* Recognising Number Plate and saved on a Folder
```
        roi_blue = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        number_plate = number_plate.detectMultiScale(roi_blue) 
        for (nx,ny,nw,nh) in number_plate:
            number_plate = cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2) 
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Number_Plate',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA) 

            grey_image = cv2.imread(number_plate,cv2.IMREAD.GREYSCALE)
            cv2.imwrite("C:\Home\ML_Project\Number_Plates_detected",grey_image) 
```
![image](https://github.com/SachinHR/Automatic-number-plate-recognition-for-motorcyclists-riding-without-helmet./blob/master/Image/Number_plate.jpg)

* Image showing
```
    cv2.imshow('img',img)   
    k = cv2.waitKey(30) & 0xff   
    if k == 27:                 
        break
```

* Destroy all windows
```
cap.release()
cv2.destroyAllWindows()   
```

## Table of Contents
* [Description](#Description)
* [Installation](#Installation)

# Installation

## Requirements
* Python 3.3+
* macOS or Linux (Windows not officially supported, but might work)

## Installation Options:

### Installing on Mac or Linux
First, make sure you have installed Python 3.3+ on your machine.
* [Install Python](https://realpython.com/installing-python/)
* Install OpenCV package on terminal

```
 pip install opencv-python

           or

 sudo apt install python3-opencv
```



