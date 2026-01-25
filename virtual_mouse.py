import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

pTime=0           #present time
plocx,plocy=0,0   #past location of x and y
clocx,clocy=0,0   #current location of x and y 
widthCam=640      #camera width
heightCam=480     #camera height
frameR=100        #frame reduction
smooth=10

cap=cv2.VideoCapture(0)
cap.set(3,widthCam)
cap.set(4,heightCam)

detector=htm.handDetector(maxHands=1)
widthSc,heightSc=autopy.screen.size()
# print(widthSc,heightSc)

while True:

  #1.find hand landmarks
  success,Img=cap.read()
  Img=detector.findHands(Img)
  Lmlist,bbox=detector.findPosition(Img)

  #2.get the tip of the index and middle fingers 
  if len(Lmlist)!=0:
      x1,y1=Lmlist[8][1:]
      x2,y2=Lmlist[12][1:]
  
      #3.check which fingers are up 
      fingers=detector.fingersUp()

       #print(fingers)
      cv2.rectangle(Img,(frameR,frameR),(widthCam-frameR,heightCam-frameR),(255,0,255),2)

      #4.only index finger :moving mode
      if fingers[1]==1 and fingers[2]==0:
          
        #5. convert coordinates
        x3=np.interp(x1,(0,widthCam),(0,widthSc))
        y3=np.interp(y1,(0,heightCam),(0,heightSc))


        #6.smoothen values
        clocx=plocx+(x3-plocx)/smooth
        clocy=plocy+(y3-plocy)/smooth

        #7.move mouse
        autopy.mouse.move(widthSc-clocx,clocy)
        cv2.circle(Img,(x1,y1),15,(255,0,255),cv2.FILLED)
        plocx,plocy=clocx,clocy
         
      #8.check if clicking mode i.e. both index and middle finger are up
      if fingers[1]==1 and fingers[2]==1:
          
          #9.find distance between fingers
          lenght,Img,lineinfo=detector.findDistance(8,12,Img)
          print(lenght)

          #10. click mouse if distance short
          if lenght<40:
              cv2.circle(Img,(lineinfo[4],lineinfo[5]),15,(0,255,0),cv2.FILLED)
              autopy.mouse.click()
      
  #11. frame rate
  cTime=time.time()
  fps=1/(cTime-pTime)
  pTime=cTime
  cv2.putText(Img ,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
  
  
  #12.display
  cv2.imshow("Image",Img)
   # PRESS 'esc' TO EXIT
  if cv2.waitKey(1)== 27:
        break
cap.release()
cv2.destroyAllWindows()
