import cv2
import numpy as np


def cropBorders(img, l=0.01, r=0.01, u=0.04, d=0.04):
    '''
        Delete bright white borders / corners.
    '''
    nrows, ncols = img.shape

    # Get the start and end rows and columns
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1 - r))
    u_crop = int(nrows * u)
    d_crop = int(nrows * (1 - d))

    cropped_img = img[u_crop:d_crop, l_crop:r_crop]

    return cropped_img

def minMaxNormalise(img):

    norm_img = (img - img.min()) / (img.max() - img.min())

    return norm_img

def globalBinarise(img, thresh, maxval): #logger,


    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval

    #ret, binarised_img = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)

    return binarised_img

def editMask(mask, ksize=(23, 23), operation="open"): #logger,

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)

    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Then dilate
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

    return edited_mask

def sortContoursByArea(contours, reverse=True): #logger,

    # Sort contours based on contour area.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)

    # Construct the list of corresponding bounding boxes.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

    return sorted_contours, bounding_boxes

def xLargestBlobs(mask, top_x=None, reverse=True): #logger,

    # Find all contours from binarised image.
    # Note: parts of the image that you want to get should be white.
    contours, hierarchy = cv2.findContours(
        image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )

    n_contours = len(contours)

    # Only get largest blob if there is at least 1 contour.
    if n_contours > 0:

        # Make sure that the number of contours to keep is at most equal
        # to the number of contours present in the mask.
        if top_x == None or n_contours < top_x:
            top_x = n_contours

        # Sort contours based on contour area.
        sorted_contours, bounding_boxes = sortContoursByArea(
            contours=contours, reverse=reverse
        )

        # Get the top X largest contours.
        X_largest_contours = sorted_contours[0:top_x]

        # Create black canvas to draw contours on.
        to_draw_on = np.zeros(mask.shape, np.uint8)

        # Draw contours in X_largest_contours.
        X_largest_blobs = cv2.drawContours(
            image=to_draw_on,  # Draw the contours on `to_draw_on`.
            contours=X_largest_contours,  # List of contours to draw.
            contourIdx=-1,  # Draw all contours in `contours`.
            color=1,  # Draw the contours in white.
            thickness=-1,  # Thickness of the contour lines.
        )

        return n_contours, X_largest_blobs
    else:
        raise ValueError("Contours must be greater than 0.")

def applyMask(img, mask): #logger,

    masked_img = img.copy()
    masked_img[mask == 0] = 0

    return masked_img

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

def clahe(img, clip=2.0, tile=(8, 8)):

    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img

def pad(img):

    nrows, ncols = img.shape

    # If padding is required...
    if nrows != ncols:

        # Take the longer side as the target shape.
        if ncols < nrows:
            target_shape = (nrows, nrows)
        elif nrows < ncols:
            target_shape = (ncols, ncols)

        # pad.
        padded_img = np.zeros(shape=target_shape)
        padded_img[:nrows, :ncols] = img

    # If padding is not required...
    elif nrows == ncols:

        # Return original image.
        padded_img = img

    return padded_img

def fullMammoPreprocess(img, l, r, d, u, thresh, maxval, ksize, operation, reverse, top_x, clip, tile):
    '''
        def fullMammoPreprocess(img, l, r, d, u, thresh, maxval, ksize, operation, reverse, top_x, clip, tile):
    '''
    cv2.imshow("img", img)
    cv2.waitKey(0)
    # Step 1: Initial crop.
    cropped_img = cropBorders(img=img, l=l, r=r, d=d, u=u)
    cv2.imshow("img", cropped_img)
    cv2.waitKey(0)
    # Step 2: Min-max normalise.
    norm_img = minMaxNormalise(img=cropped_img)
    cv2.imshow("img", norm_img)
    cv2.waitKey(0)
    # Step 3: Remove artefacts.
    binarised_img = globalBinarise(img=norm_img, thresh=thresh, maxval=maxval)
    edited_mask = editMask(
        mask=binarised_img, ksize=(ksize, ksize), operation=operation
    )
    cv2.imshow("img", binarised_img)
    cv2.waitKey(0)
    _, xlargest_mask = xLargestBlobs(mask=edited_mask, top_x=top_x, reverse=reverse)

    masked_img = applyMask(img=norm_img, mask=xlargest_mask)
    cv2.imshow("img", masked_img)
    cv2.waitKey(0)
    # Step 4: Horizontal flip.
    lr_flip = checkLRFlip(mask=xlargest_mask)
    if lr_flip:
        flipped_img = makeLRFlip(img=masked_img)
    elif not lr_flip:
        flipped_img = masked_img
    cv2.imshow("flipped_img", flipped_img)
    cv2.waitKey(0)
    # Step 5: CLAHE enhancement.
    clahe_img = clahe(img=flipped_img, clip=clip, tile=(tile, tile))
    cv2.imshow("img", masked_img)
    cv2.waitKey(0)
    # Step 6: pad.
    padded_img = pad(img=clahe_img)
    cv2.imshow("padded_img", padded_img)
    cv2.waitKey(0)
    padded_img = cv2.normalize(
        padded_img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    # Step 7: Min-max normalise.
    img_pre = minMaxNormalise(img=padded_img)
    cv2.imshow("img_pre", img_pre)
    cv2.waitKey(0)

    return img_pre, lr_flip

img = cv2.imread("C:/Users/RDuser-E1/Desktop/TFG/all-mias/images/mdb312.pgm", 0)

img_pre, lr_flip = fullMammoPreprocess(img=img, l=0.01, r=0.01, d=0.04, u=0.04, thresh=0.1, maxval=255, ksize=23, operation="open", reverse=True, top_x=None, clip=2.0, tile=(8, 8))

cv2.imshow('imagen',img_pre)
cv2.waitKey(0)