import cv2
import os
path_to_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/images/"
path_to_croped_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/croped_images/"
files_names = os.listdir(path_to_images)

#Código propio

for image_name in files_names:
    # Read the image:
    image= cv2.imread(path_to_images + image_name, 0)

    # Search limits of borders:
    x1 = 0
    while image[512,x1] == 0:
            x1 += 1
    print(x1)

    x2 = 1023
    while image[512, x2] == 0:
        x2 -= 1
    print(x2)
    
    # crop:
    croped_image = image[0:1023, x1:x2]
    
    # Save as png:
    image_name_as_png = image_name.replace("pgm", "png")
    cv2.imwrite(path_to_croped_images + image_name_as_png, croped_image)

#cv2.imshow("mamografía", croped_image)
#cv2.waitKey(0)
