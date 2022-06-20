import pandas as pd
import cv2
import numpy as np
path_to_data = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/info/MIAS.csv"
path_to_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/croped_images/"

data = pd.read_csv(path_to_data)

# Extract from data:
tissue = np.array(data['tissue'])
reference = data['reference']
images_name = np.array(reference + '.png')

print(images_name)

max_Fatty = []
min_Fatty = []

max_Fatty_Glandular = []
min_Fatty_Glandular = []

max_Dense_Glandular = []
min_Dense_Glandular = []

cnt = 0 # Loop in tissue

for image_name in images_name:
    image = cv2.imread(path_to_images + image_name, 0)
    max = np.amax(image)
    min = np.amin(image)

    if tissue[cnt] == 'F':

        max_Fatty.append(max)
        min_Fatty.append(min)

    elif tissue[cnt] == 'G':

        max_Fatty_Glandular.append(max)
        min_Fatty_Glandular.append(min)

    elif tissue[cnt] == 'D':

        max_Dense_Glandular.append(max)
        min_Dense_Glandular.append(min)

    else:
        raise ValueError(tissue[cnt], " is not a valid type of tissue.")

    cnt += 1

# Convert to numpy array results:

max_Fatty = np.array(max_Fatty)
print(max_Fatty)
min_Fatty = np.array(min_Fatty)
print(min_Fatty)

max_Fatty_Glandular = np.array(max_Fatty_Glandular)
print(max_Fatty_Glandular)
min_Fatty_Glandular = np.array(min_Fatty_Glandular)
print(min_Fatty_Glandular)

max_Dense_Glandular = np.array(max_Dense_Glandular)
print(max_Dense_Glandular)
min_Dense_Glandular = np.array(min_Dense_Glandular)
print(min_Dense_Glandular)

print("Fatty(F): ")
print("max: ", np.amax(max_Fatty))
print("mean max: ", np.mean(max_Fatty))
print("min: ", np.amin(min_Fatty))
print("Fatty-glandular(G): ")
print("max: ", np.amax(max_Fatty_Glandular))
print("mean max: ", np.mean(max_Fatty_Glandular))
print("min: ", np.amin(min_Fatty_Glandular))
print("Dense-glandular(D): ")
print("max: ", np.amax(max_Dense_Glandular))
print("mean max: ", np.mean(max_Dense_Glandular))
print("min: ", np.amin(min_Dense_Glandular))




