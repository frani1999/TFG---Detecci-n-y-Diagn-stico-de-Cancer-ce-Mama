import cv2
import os

path_to_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/anotations_removed_preprocessing_stack/pgm/"
path_to_jpg_png_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/anotations_removed_preprocessing_stack/jpg/"
files_names = os.listdir(path_to_images)

for image_name in files_names:

    img = cv2.imread(path_to_images + image_name)

    cv2.imwrite(path_to_jpg_png_images + image_name.replace("pgm", "jpg"), img)