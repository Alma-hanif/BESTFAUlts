import cv2
import numpy as np

img = cv2.imread(r'D:\Work\INEA_EdgeChips\Data\20200221_133005_381.ply.png')
image = cv2.resize(img, (500, 500))
image_contours = np.zeros((img.shape[1], img.shape[0], 1), np.uint8)

image_binary = np.zeros((img.shape[1], img.shape[0], 1), np.uint8)

for channel in range(img.shape[2]):
    ret, image_thresh = cv2.threshold(img[:, :, channel], 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(image_thresh, 1, 1)[0]
    cv2.drawContours(image_contours, contours, -1, (255,255,255), 3)

contours = cv2.findContours(image_contours, cv2.RETR_LIST,
                           cv2.CHAIN_APPROX_SIMPLE)[0]

cv2.drawContours(image_binary, [max(contours, key = cv2.contourArea)],
                -1, (255, 255, 255), -1)

cv2.imwrite('Output/fill_contour.jpg', image_binary)
cv2.imshow('Small Contour', image_binary)
cv2.waitKey(0)