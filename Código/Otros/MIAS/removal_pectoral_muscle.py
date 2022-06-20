import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab as pylab
from skimage import io
from skimage import color
import cv2
from skimage.feature import canny
from skimage.filters import sobel
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import polygon
import os
# https://github.com/gsunit/Pectoral-Muscle-Removal-From-Mammograms

'''
def right_orient_mammogram(image):
    left_nonzero = cv2.countNonZero(image[:, 0:int(image.shape[1] / 2)])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1] / 2):])

    if (left_nonzero < right_nonzero):
        image = cv2.flip(image, 1)

    return image
'''
def read_image(filename):
    image = io.imread(filename, as_gray=True)
    #image = color.rgb2gray(image)
    #image = right_orient_mammogram(image)
    return image

def apply_canny(image):
    canny_img = canny(image, 1)
    return sobel(canny_img)

def get_hough_lines(canny_img):
    h, theta, d = hough_line(canny_img)
    lines = list()
    print('\nAll hough lines')
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        print("Angle: {:.2f}, Dist: {:.2f}".format(np.degrees(angle), dist))
        x1 = 0
        y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
        x2 = canny_img.shape[1]
        y2 = (dist - x2 * np.cos(angle)) / np.sin(angle)
        lines.append({
            'dist': dist,
            'angle': np.degrees(angle),
            'point1': [x1, y1],
            'point2': [x2, y2]
        })

    return lines

def shortlist_lines(lines):
    MIN_ANGLE = 10
    MAX_ANGLE = 70
    MIN_DIST = 5
    MAX_DIST = 200

    shortlisted_lines = [x for x in lines if
                         (x['dist'] >= MIN_DIST) &
                         (x['dist'] <= MAX_DIST) &
                         (x['angle'] >= MIN_ANGLE) &
                         (x['angle'] <= MAX_ANGLE)
                         ]
    print('\nShorlisted lines')
    for i in shortlisted_lines:
        print("Angle: {:.2f}, Dist: {:.2f}".format(i['angle'], i['dist']))

    return shortlisted_lines

def remove_pectoral(shortlisted_lines):
    shortlisted_lines.sort(key=lambda x: x['dist'])
    if len(shortlisted_lines) != 0:
        pectoral_line = shortlisted_lines[0]
        d = pectoral_line['dist']
        theta = np.radians(pectoral_line['angle'])

        x_intercept = d / np.cos(theta)
        y_intercept = d / np.sin(theta)

        return  polygon([0, 0, y_intercept], [0, x_intercept, 0])
    else:
        return polygon([0, 0, 0], [0, 0, 0])

def display_image(filename, filesave, filepng):
    image = read_image(filename)
    #print(image.shape[0])
    #print(image.shape[1])
    canny_image = apply_canny(image)
    lines = get_hough_lines(image)
    shortlisted_lines = shortlist_lines(lines)

    fig, axes = plt.subplots(1, 4, figsize=(15, 10))
    fig.tight_layout(pad=3.0)
   # plt.xlim(0, image.shape[1])
    #plt.ylim(image.shape[0])

    #axes[0].set_title('Right-oriented mammogram')
    #axes[0].imshow(image, cmap=pylab.cm.gray)
    #axes[0].axis('on')

    #axes[1].set_title('Hough Lines on Canny Edge Image')
    #axes[1].imshow(canny_image, cmap=pylab.cm.gray) # canny_image
    #axes[1].axis('on')
    #axes[1].set_xlim(0, image.shape[1])
    #axes[1].set_ylim(image.shape[0])
    #for line in lines:
        #axes[1].plot((line['point1'][0], line['point2'][0]), (line['point1'][1], line['point2'][1]), '-r')

    #axes[2].set_title('Shortlisted Lines')
    #axes[2].imshow(canny_image, cmap=pylab.cm.gray) #canny_image
    #axes[2].axis('on')
    #axes[2].set_xlim(0, image.shape[1])
    #axes[2].set_ylim(image.shape[0])
    #for line in shortlisted_lines:
        #axes[2].plot((line['point1'][0], line['point2'][0]), (line['point1'][1], line['point2'][1]), '-r')

    rr, cc = remove_pectoral(shortlisted_lines) #  rr, cc
    rr[np.where(rr >= 943)] = 942
    #rr_unique = np.unique(rr)
    #cc_unique = np.unique(cc)
    image[rr, cc] = 0 #[:image.shape[0]]
    #axes[3].set_title('Pectoral muscle removed')
    #axes[3].imshow(image, cmap=pylab.cm.gray)
    #axes[3].axis('on')

    #plt.show()

    cv2.imwrite(filesave, image)
    cv2.imwrite(filepng, image)

path_to_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/anotations_removed_preprocessing_stack/"
path_to_removal_pectoral_muscle_image = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/pectoral_muscle_removed/"
path_to_removal_pectoral_muscle_image_png = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/pectoral_muscle_removed/png/"

files_names = os.listdir(path_to_images)
files_names.remove("anotations_removed_preprocessing_stack_as_png")

for image_name in files_names:
    display_image(path_to_images + image_name, path_to_removal_pectoral_muscle_image + image_name, path_to_removal_pectoral_muscle_image_png + image_name.replace("pgm", "png"))


# "C:/Users/RDuser-E1/Desktop/TFG/all-mias/mammo_1.png"
