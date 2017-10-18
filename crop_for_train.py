import cv2
import numpy as np
import re
import os
import random

TRAINING_SET_DIRECTORY = "Contest\\"
PROCESSED_DIRECTORY = "Processed\\"
CHICKY_SET_DIRECTORY = "Chicky\\"
CHICKY_PROCESSED_DIRECTORY = "ChickyProcessed\\"
LIST_FILENAME = TRAINING_SET_DIRECTORY + "list.txt"
RESULT_FILENAME = TRAINING_SET_DIRECTORY + "result.txt"
SAVE_IMAGE_EXTENSION = '.jpg'

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + x1
    y2 = boxes[:, 3] + y1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]

def SaveImage(image, string):
    if CHICKY:
        cv2.imwrite(CHICKY_PROCESSED_DIRECTORY + CURRENT_FILENAME + '-' + string + SAVE_IMAGE_EXTENSION, image)
    else:
        cv2.imwrite(PROCESSED_DIRECTORY + CURRENT_FILENAME + '-' + string + SAVE_IMAGE_EXTENSION, image)

# Read filename list
with open(LIST_FILENAME) as f:
    training_set_list = f.readlines()
training_set_list = [x.strip() for x in training_set_list if x.strip()]

for IMAGE_FILENAME in os.listdir(CHICKY_SET_DIRECTORY):
    CHICKY = 1
    # for IMAGE_FILENAME in training_set_list:
    centerBoundingBoxes = []
    fileName = re.split('\.', IMAGE_FILENAME)
    CURRENT_FILENAME = fileName[0]
    if CHICKY:
        im_original = cv2.imread(CHICKY_SET_DIRECTORY + IMAGE_FILENAME)
    else:
        im_original = cv2.imread(TRAINING_SET_DIRECTORY + IMAGE_FILENAME)
    im_for_crop = im_original.copy()
    im = adjust_gamma(im_original, 1.9)
    im = cv2.GaussianBlur(im, (3, 3), 0)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(hsv)
    # s.fill(155)
    # v.fill(255)
    hsv_image = cv2.merge([h, s, v])
    LOWER_YELLOW = np.array([16, 140, 110], dtype=np.uint8)
    UPPER_YELLOW = np.array([95, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
    closing_se = np.ones((50, 50), np.uint8)
    im = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, closing_se)
    # closing_se = np.ones((30, 30), np.uint8)
    # im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, closing_se)
    closing_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, closing_se)
    # opening_se = np.ones((5, 5), np.uint8)
    # im = cv2.morphologyEx(im, cv2.MORPH_OPEN, opening_se)
    # erode_se = np.ones((1, 1), np.uint8)  # structuring element
    # im = cv2.erode(yellow_mask, erode_se, iterations=2)
    im = cv2.medianBlur(im, 5)
    yellow_mask = im.copy()
    cv2.imshow(CURRENT_FILENAME + ' yellow mask (binary)', yellow_mask)
    yellow_res = cv2.bitwise_and(im_original, im_original, mask=yellow_mask)
    # cv2.imshow(IMAGE_FILENAME + "bitwise", yellow_res)

    contourmask = yellow_mask
    temp, contours, hierarchy = cv2.findContours(contourmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundingBoxes = []
    moments = []
    for cnt in contours:
        bRect = cv2.boundingRect(cnt)
        boundingBoxes.append(bRect)
        moments.append(cv2.moments(cnt))
    boundingBoxes = non_max_suppression_slow(np.array(boundingBoxes), 0.2)
    okBoundingBoxes = []
    okNumYellowPixels = []
    numBoxes = 0
    for (x, y, w, h) in boundingBoxes:
        numBoxes = numBoxes + 1
        im_window = im_original[y:(y+h), x:(x+w)]
        im_window = adjust_gamma(im_window, 1.9)
        if im_window.size < 0.011 * im_original.size:
            continue
        hsv_image = cv2.cvtColor(im_window, cv2.COLOR_BGR2HSV_FULL)
        LOWER_RED1 = np.array([170, 40, 50], dtype=np.uint8)
        UPPER_RED1 = np.array([180, 255, 255], dtype=np.uint8)
        LOWER_RED2 = np.array([0, 40, 50], dtype=np.uint8)
        UPPER_RED2 = np.array([15, 255, 255], dtype=np.uint8)
        hsv_image = cv2.medianBlur(hsv_image, 5)
        red_mask1 = cv2.inRange(hsv_image, LOWER_RED1, UPPER_RED1)
        red_mask2 = cv2.inRange(hsv_image, LOWER_RED2, UPPER_RED2)
        red_mask = red_mask1 + red_mask2
        cv2.imshow(CURRENT_FILENAME + " red_mask (" + str(x) + ", " + str(y) + ", " + str(w) + ", " + str(h) + ")", red_mask)
        LOWER_YELLOW = np.array([29, 140, 110], dtype=np.uint8)
        UPPER_YELLOW = np.array([48, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
        cv2.imshow(CURRENT_FILENAME + " yellow_mask (" + str(x) + ", " + str(y) + ", " + str(w) + ", " + str(h) + ")", yellow_mask)
        num_red_pixels = cv2.countNonZero(red_mask)
        num_yellow_pixels = cv2.countNonZero(yellow_mask)
        if (num_yellow_pixels > num_red_pixels) \
                and (num_red_pixels > 0.0006 * im_window.size) \
                and (num_red_pixels < 0.2 * im_window.size):
            box = (x, y, w, h)
            okBoundingBoxes.append(box)
            okNumYellowPixels.append(num_yellow_pixels)
        cv2.rectangle(im_original, (x, y), (x + w, y + h), (255, 0, 0), 2)
        M = moments[numBoxes - 1]
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cropped_center = im_for_crop[(cY-25):(cY+25),(cX-25):(cX+25)]
            cropped_file_name = "ChickyTextures//" + CURRENT_FILENAME + '-' + str(random.randrange(10)) + SAVE_IMAGE_EXTENSION
            if cropped_center.size > 0:
                cv2.imwrite(cropped_file_name, cropped_center)
                cv2.imshow(cropped_file_name, cropped_center)

    print(CURRENT_FILENAME + ": #boxes = " + str(numBoxes))
    if okNumYellowPixels:
        okIdx = np.argmax(np.array(okNumYellowPixels))
        (x, y, w, h) = okBoundingBoxes[okIdx]
        cv2.rectangle(im_original, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.drawContours(im, contours, -1, (0, 255, 0), 2)
    # cv2.imshow(IMAGE_FILENAME, im_original)
    SaveImage(im_original, 'result')

    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('w'):
    #         break
    # cv2.destroyAllWindows()
    # exit()
