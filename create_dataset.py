import cv2
import os

exit_con='**'

a=''
dir0 = input('Enter the directory name: ')

try:
  os.mkdir(dir0)
except:
  print('duplicate directory name!')

camera = cv2.VideoCapture(0)
while(True):
  a = input('exit: ** or enter the label name : ')
  if(a==exit_con):
    break
  dir1 = str(dir0)+'/'+str(a)
  print(dir1)

  try:
    os.mkdir(dir1)
  except:
    print('contain folder')

  j = 0
  while(j<300):
    try:
      os.mkdir(str(dir1)+'/'+str(j))
      j+=1
    except:
      print('contain folder')

  i = 0
  time = 1
  k = 0
  while(True):
    (t, frame) = camera.read() 

    #flip frame so that it's not the mirror

    frame = cv2.flip(frame, 1)

    #reduce noise
    # img = cv2.GaussianBlur(frame, (7,7),0)
    if(i%4==0):
      cv2.imwrite("%s/%s/%d/%d.jpg"%(dir0, a, k, time), frame)
      time+=1

    i+=1
    print(i)

    if(i>36*4):
      k+=1
      i = 0
      time = 1
    if(k==300): break
    cv2.imshow("Video Feed:", frame)

    keypress = cv2.waitKey(1)

    #escape
    if keypress == 27: 
      break

camera.release()
cv2.destroyAllWindows()


