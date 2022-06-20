import cv2
import numpy as np

# read image
img = cv2.imread("C:/Users/RDuser-E1/Desktop/TFG/all-mias/images/mdb31.pgm")
hh, ww = img.shape[:2]

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply otsu thresholding
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1]

# apply morphology close to remove small regions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# apply morphology open to separate breast from other regions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

# get largest contour
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)

# draw largest contour as white filled on black background as mask
mask = np.zeros((hh,ww), dtype=np.uint8)
cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

# dilate mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55))
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

# apply mask to image
result = cv2.bitwise_and(img, img, mask=mask)

# save results
#cv2.imwrite('mammogram_thresh.jpg', thresh)
#cv2.imwrite('mammogram_morph2.jpg', morph)
#cv2.imwrite('mammogram_mask2.jpg', mask)
#cv2.imwrite('mammogram_result2.jpg', result)

# show resultls
cv2.imshow("original", img)
cv2.imshow('thresh', thresh)
cv2.imshow('morph', morph)
cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()