import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\Work\INEA_EdgeChips\Data\20200221_133005_381.ply.png', cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (500, 500))
print(img)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,55,
                            param1=60,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
