import cv2

motor_cycle = cv2.CascadeClassifier('haarcascade_motorcyclist_without_helmet.xml')                          # Assigning XML file
number_plate = cv2.CascadeClassifier('haarcascade_number_plate.xml')

cap = cv2.VideoCapture(0)                                                                                   # Capture Video 

while 1: 
    ret, img = cap.read()                                                                                   # Reading Live Video
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    motor_cycle = motor_cycle.detectMultiScale(gray, 1.3, 5)                                                # Detecting MotorCyclistWithoutHelmet

    for (x,y,w,h) in motor_cycle:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)                                                      # Marking out a rectangle on MotorCycle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Motor_Cyclist_Without_Helmet',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)  # Put text on MotorCycle Rectangle 

        roi_red = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        number_plate = number_plate.detectMultiScale(roi_red)                                               # Detecting MotorCycle Number Plates
        for (nx,ny,nw,nh) in number_plate:
            number_plate = cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)                       # Marking out a rectangle on number plates
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Number_Plate',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)              # Put text on No. Plate Rectangle


            grey_image = cv2.imread(number_plate,cv2.IMREAD.GREYSCALE)
            cv2.imwrite("C:\Home\ML_Project\Number_Plates_detected",grey_image)                             # Saving Number Plates on a Folder


    cv2.imshow('img',img)                                                                                   # Showing Image
    k = cv2.waitKey(30) & 0xff                                                                              # for exit
    if k == 27:                    
        break

cap.release()
cv2.destroyAllWindows()                                                                                     # Destroying all windows
