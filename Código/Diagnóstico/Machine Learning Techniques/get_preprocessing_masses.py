import cv2
import pandas as pd
import os
import skimage
from scipy.ndimage import binary_fill_holes

#Path to train data:
path_train = "C:/Users/RDuser-E1/Desktop/TFG/Diagnosis/masses/image_crop_masses_train.csv"
#Path to test data:
path_test = "C:/Users/RDuser-E1/Desktop/TFG/Diagnosis/masses/image_crop_masses_test.csv"

#Path to result images:
path_to_output_images = "T:/fbr/CBIS-DDSM/preprocessing_masses/"

output = pd.DataFramerame()
output['image_name'] = []



#TRAIN DATA:

print("\n\n\n\n---------TRAIN DATA---------\n\n\n\n")

#Extract route to images:
train_data = pd.read_csv(path_train)
cropped_image_folder = train_data['image_folder']
cropped_image_name = train_data['image_name']

#Iterate: read, process and store each image:
for i in range(len(cropped_image_folder)):
    print(cropped_image_folder[i] + "/" + cropped_image_name[i])

    img = cv2.imread("T:/fbr/CBIS-DDSM/jpeg/" + cropped_image_folder[i] + "/" + cropped_image_name[i], 0)

    # Pre_process:
    # Normalize image:
    img = img / 255
    # Apply gaussian filter:
    img_gaus = (cv2.GaussianBlur(img, (5, 5), 0) * 255).astype('uint8')
    # Apply threshold:
    t, mask = cv2.threshold(img_gaus, 0, 255, cv2.THRESH_OTSU)
    # Invert:
    mask_norm = ((255 - mask) / 255) == 0
    # Remove artifacts:
    img_removed = skimage.morphology.remove_small_objects(mask_norm, min_size=64)
    # Fill lumens objects:
    mask2 = binary_fill_holes(img_removed)
    # Resize for addapt to input of model:
    #img_resize = cv2.resize((mask2).astype('uint8'), (224, 224))

    #Store output image:
    cv2.imwrite(path_to_output_images + cropped_image_folder[i] + "_" + cropped_image_name[i], (mask2 * 255).astype('uint8'))


#TEST DATA:

print("\n\n\n\n---------TEST DATA---------\n\n\n\n")

#Extract route to images:
test_data = pd.read_csv(path_test)
cropped_image_folder = test_data['image_folder']
cropped_image_name = test_data['image_name']

#Iterate: read, process and store each image:
for i in range(len(cropped_image_folder)):
    print(cropped_image_folder[i] + "/" + cropped_image_name[i])

    img = cv2.imread("T:/fbr/CBIS-DDSM/jpeg/" + cropped_image_folder[i] + "/" + cropped_image_name[i], 0)

    # Pre_process:
    # Normalize image:
    img = img / 255
    # Apply gaussian filter:
    img_gaus = (cv2.GaussianBlur(img, (5, 5), 0) * 255).astype('uint8')
    # Apply threshold:
    t, mask = cv2.threshold(img_gaus, 0, 255, cv2.THRESH_OTSU)
    # Invert:
    mask_norm = ((255 - mask) / 255) == 0
    # Remove artifacts:
    img_removed = skimage.morphology.remove_small_objects(mask_norm, min_size=64)
    # Fill lumens objects:
    mask2 = binary_fill_holes(img_removed)
    # Resize for addapt to input of model:
    #img_resize = cv2.resize((mask2).astype('uint8'), (224, 224))

    #Store output image:
    cv2.imwrite(path_to_output_images + cropped_image_folder[i] + "_" + cropped_image_name[i], (mask2 * 255).astype('uint8'))

