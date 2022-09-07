import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread(r"D:\\seam-carving-master\\in\\images\\1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.GaussianBlur(img,(5,5), 0)
gray_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l_mean = np.mean(gray_lab[:,:,0])
a_mean = np.mean(gray_lab[:,:,1])
b_mean = np.mean(gray_lab[:,:,2])
lab = np.square(gray_lab- np.array([l_mean, a_mean, b_mean]))
lab = np.sum(lab,axis=2)
lab = lab/np.max(lab)
lab*=256
lab=lab.astype(np.uint8)
ret, lab = cv2.threshold(lab, 150, 255, cv2.THRESH_BINARY)
cv2.imwrite("D:\\seam-carving-master\\in\\masks\\mask.jpg", lab)

plt.imshow(lab, cmap='gray')
plt.show()
