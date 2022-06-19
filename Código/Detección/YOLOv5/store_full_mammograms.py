import pandas as pd
import cv2

#Paths:
path_to_data = "full_mammogram_images.csv"
path_to_images = "T:/fbr/CBIS-DDSM/jpeg/"
path_to_output_images = "T:/fbr/CBIS-DDSM/full_mammograms_images/"

#Read csv file:
data = pd.read_csv(path_to_data)

#Extract data from csv file:
image_path = data["image_path"]
image_name = data["image_name"]


cnt = 0
for image in image_name:
    #Show name of image:
    print(path_to_images + image_path[cnt] + "/" + image)

    #Read image:
    img = cv2.imread(path_to_images + image_path[cnt] + "/" + image)

    #Store image:
    #print(path_to_output_images + image_path[cnt] + "_" + image)
    cv2.imwrite(path_to_output_images + image_path[cnt] + "_" + image, img)

    cnt += 1
