import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
frame = cv2.imread(r"D:\\seam-carving-master\\in\\images\\3.jpg")
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor = 1.15,#缩放比
    minNeighbors = 10,#敏感度
    minSize = (5,5),
)
print ("发现{0}个人脸!".format(len(faces)))
frame[:,:]=(0,0,0)
for(x,y,w,h) in faces:
    cv2.circle(frame, (int(x+w/2),int(y+h/2)), int(w/2), (255,255,255), -1)
    
    gray[y:y+h, x:x+w]=255
    face_area_draw = frame[y:y+h, x:x+w]
ret, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
   
cv2.imshow("f", frame)
cv2.imshow("gray", gray)
cv2.imwrite("D:\\seam-carving-master\\in\\masks\\mask.jpg", frame)
cv2.waitKey(0)  