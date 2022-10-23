import cv2
import numpy as np

cap = cv2.VideoCapture(0)
flag = cap.isOpened()

if not flag:
    print("Cannot open camera")
    exit()

ret, img = cap.read()
cap.release()
img_cont = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_resize = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
canny_1 = 200
canny_2 = 225
canny = cv2.Canny(img, canny_1, canny_2)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
min_black = 255
cnt_black = []

for cnt in contours:
    c_area = cv2.contourArea(cnt) + 1e-7
    if cv2.contourArea(cnt) + 1e-7 > 500:
        cv2.drawContours(img_cont, [cnt], -1, 3)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, (255,255,255), -1)
        temp_mask = cv2.bitwise_and(gray, mask)
        temp_col = np.sum(temp_mask).real/(cv2.contourArea(cnt)+1e-7)
        if (temp_col < min_black) or (len(cnt_black) == 0):
            cnt_black = cnt
            min_black = temp_col

if len(cnt_black)!=0:
    cv2.drawContours(img_cont, [cnt_black], -1, (0,0,255), 3)

cv2.imwrite('img_basic.jpg', img)
cv2.imwrite('img_contour.jpg', img_cont)
cv2.imwrite('img_gray_channel.png', gray)
cv2.imwrite('img_resize.png', img_resize)

cv2.imshow('Basic image', img)
cv2.imshow('Contour image', img_cont)
cv2.imshow('Gray channel image', gray)
cv2.imshow('Resize image', img_resize)
cv2.waitKey(0)