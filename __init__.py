import numpy as np #this library support n-dimension array, use to store image array for opencv
import cv2   #opencv object 
import time  #time counter object
import winsound #sound lib object
import threading #multhread lib object, use to play the wav sound
from time import sleep #this object diect call sleep library out for use
import datetime # this the log to indicate the date and time of the output in text

playSound = False # set global play sound flag to false
exitLoop = False # set thread loop exit flag to false

# create loop function to call in thread
def playAlertSound():
    # mention these variables are global from declaration above
    global playSound 
    global exitLoop
    
    while 1:
        #if loop flag true then exit the thread loop
        if exitLoop:
            break;
        #if play sound flag true then play the wav soun
        if playSound:
            #after play then set flag to false, to avoid sound keep playing when yawning is detected
            playSound = False
            winsound.PlaySound('AlarmDrum.wav', winsound.SND_NOSTOP)
        
#load face and eye cascade into cv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')

#load usb camera 
cap = cv2.VideoCapture(0)

#counter variable initial set to 0
yawmingTimeCounter = 0

# start the play sound thread when yanwing is detected then global flag playSound will set to true to play the sound
# sound is playing in another thread is because if sound playing in same thread within the while loop for capturing frame from camera, 
# when yawning is detected then ound will start play but next process will need to waiting the sound finish playing then only process fro next frame
t = threading.Thread(target=playAlertSound)
# start the sound loop thread
t.start()

#read frame from camera and onvert it to gray
while 1:
    # set yawning flag to false for every new frame prcoess
    yawning = False
    
    # read fram from camera captured
    ret, img = cap.read()
    
    # convert image capture from color to grayscale
    # face detection is required grayscale color, so this step must do it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #pass grayscale image to face cascade and with the threshold of 1.3 
    #cascade is function use to perform face detection but required pre trained face xml
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #draw the face found from cascade on frame  
    #trace into rectangle found in face cascade 
    for (x,y,w,h) in faces:
        # rectangle found are face detected, then draw it on from by rectangle
        cv2.rectangle(img,(x,y),(x+w,y+h+50),(255,0,0),2)
        
        # increase the height of face found by 50
        # sometimes rectangle only get above mouth, if without adding 50 then sometimes cannot find the mouth from region crop out
        ymax = y+h+50
        
        # check if the increment exceed to image height 
        # if reached then set the max of height 
        if ymax > y+h:
            ymax = y+h
            
        #crop the possible mouth are from face and pass it to mouth cascade for detecting
        roi_gray = gray[y+3*h/5:ymax, x:x+w]
        roi_color = img[y+3*h/5:ymax, x:x+w]
        
        # eye detect current not use 
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        # draw rect for eye if found
        #for (ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        # find mounth detected position from face crop region
        # using threadhold of 1.3
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.3, 20)
        
        # initial required varibles
        largestX = 0
        largestY = 0
        largestW = 0
        largestH = 0
        largestArea = 0
        
        # find the largest contour found in bottom portion of face,the largest area is mouth
        for (ex,ey,ew,eh) in mouth:
            area = ew * eh
            if area > largestArea:
                largestX = ex
                largestY = ey
                largestW = ew
                largestH = eh
        
        #if contour of mouth found greather than o for width and height then check the width of mouth
        if  largestW > 0 and largestH > 0:  
            # draw mouth detected on frame
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 
            #if width of mouse less than define threshold start counter the time
            if largestW < (0.28 * w): 
                yawmingTimeCounter += 30
            else:
                yawmingTimeCounter = 0
    
    # IF time contour more than 66 times which is 1980 millisecond (30 x 66), multiple by 30 is because camera process wait time is 30 millisecond 
    if yawmingTimeCounter > 66:
        cv2.putText(img,"Yawning!!!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        playSound = True
      # if the mouth open then there is fatigue
        text_file = open("Output.txt", "w")
        text_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+ "FATIGUE")
        text_file.close()
        
    #show image
    cv2.imshow('img',img)
    
    #quit the syste by press ESC keys
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# set play sound loop flag to true so play sound threed will exited
exitLoop = True

# closing the camera device
cap.release()

# clear opencv object created 
cv2.destroyAllWindows()
