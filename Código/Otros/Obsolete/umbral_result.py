import cv2

path_to_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/croped_images/"

img = cv2.imread(path_to_images + "mdb037.png", 0)
#img = cv2.equalizeHist(img)
ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
print(ret)
cv2.imshow('image',img)
cv2.waitKey(0)