import os
import cv2

path_to_images = "T:/fbr/MIASDBv1.21/pgm_images/"
path_to_png_images = "T:/fbr/MIASDBv1.21/png_images/"

images_names = os.listdir(path_to_images)

print(images_names)

for image_name in images_names:
    #Read image:
    image = cv2.imread(path_to_images + image_name)

    # Save as png:
    image_name_as_png = image_name.replace("pgm", "png")
    cv2.imwrite(path_to_png_images + image_name_as_png, image)