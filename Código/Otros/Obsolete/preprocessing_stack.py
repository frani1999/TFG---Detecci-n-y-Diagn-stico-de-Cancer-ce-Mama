import cv2
import numpy as np
import os

def delete_vertical_borders(image):

    hh, ww = image.shape[:2]
    print(image.shape[:2])
    # Search limits of borders:
    x1 = 0
    while image[int(ww/2), x1] == 0:
        x1 += 1
    print(x1)

    x2 = hh
    while image[int(ww/2), x2] == 0:
        x2 -= 1
    print(x2)

    # crop:
    croped_image = image[0:ww, x1:x2]

    return croped_image


def cropBorders(img, l=0.01, r=0.01, u=0.04, d=0.04):
    '''
        Delete bright white borders / corners.
    '''
    nrows, ncols, channels = img.shape

    # Get the start and end rows and columns
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1 - r))
    u_crop = int(nrows * u)
    d_crop = int(nrows * (1 - d))

    cropped_img = img[u_crop:d_crop, l_crop:r_crop]

    return cropped_img


def checkLRFlip(mask):

    # Get number of rows and columns in the image.
    nrows, ncols = mask.shape
    x_center = ncols // 2
    y_center = nrows // 2

    # Sum down each column.
    col_sum = mask.sum(axis=0)
    # Sum across each row.
    row_sum = mask.sum(axis=1)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])

    if left_sum < right_sum:
        LR_flip = True
    else:
        LR_flip = False

    return LR_flip


def makeLRFlip(img):

    flipped_img = np.fliplr(img)

    return flipped_img
#-----------------------------------------------------------------------------------------------------------------------
path_to_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/images/"
path_to_processed_images = 'C:/Users/RDuser-E1/Desktop/TFG/all-mias/anotations_removed_preprocessing_stack/pgm/'
path_to_processed_png_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/anotations_removed_preprocessing_stack/png/"
path_to_processed_jpg_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/anotations_removed_preprocessing_stack/jpg/"
files_names = os.listdir(path_to_images)
print(files_names)

cnt = 1
for image_name in files_names:
    print(image_name)
    img = cv2.imread(path_to_images + image_name)

    hh, ww = img.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply otsu thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    # apply morphology close to remove small regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # apply morphology open to separate breast from other regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # apply morphology dilate to compensate for otsu threshold not getting some areas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    big_contour_area = cv2.contourArea(big_contour)

    # draw all contours but the largest as white filled on black background as mask
    mask = np.zeros((hh, ww), dtype=np.uint8)
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area != big_contour_area:
            cv2.drawContours(mask, [cntr], 0, 255, cv2.FILLED)

    # invert mask
    mask = 255 - mask

    # apply mask to image
    image_masked = cv2.bitwise_and(img, img, mask=mask)

    if cnt%2 != 0:
        image_masked = makeLRFlip(img=image_masked)

    # crop borders:
    croped_image = cropBorders(img=image_masked, l=0.01, r=0.01, d=0.04, u=0.04)
    #result = delete_vertical_borders(image=croped_image)
    # Transform result to grayscale:
    croped_image = cv2.cvtColor(croped_image, cv2.COLOR_BGR2GRAY)
    result = delete_vertical_borders(image=croped_image)
    # save results
    #cv2.imwrite('mammogram_thresh.jpg', thresh)
    #cv2.imwrite('mammogram_morph.jpg', morph)
    #cv2.imwrite('mammogram_mask.jpg', mask)
    #cv2.imwrite('mammogram_result.jpg', result)
    cv2.imwrite(path_to_processed_images + image_name, result)
    cv2.imwrite(path_to_processed_png_images + image_name.replace("pgm", "png"), result)
    cv2.imwrite(path_to_processed_jpg_images + image_name.replace("pgm", "jpg"), result)

    cnt += 1
    # show resultls

    #cv2.imshow("original", img)
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('morph', morph)
    #cv2.imshow('mask', mask)
    #cv2.imshow('result', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()