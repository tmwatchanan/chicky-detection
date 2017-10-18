import cv2
import numpy as np
import re
import os
from skimage.feature import greycomatrix, greycoprops
import random

CHICKY = 0
TRAINING_SET_DIRECTORY = "KarnContest\\"
PROCESSED_DIRECTORY = "KarnContestProcessed\\"
CHICKY_SET_DIRECTORY = "Chicky\\"
CHICKY_PROCESSED_DIRECTORY = "ChickyProcessed\\"
POSITIVE_TEXTURES_DIRECTORY = "ChickyTextures\\"
NEGATIVE_TEXTURES_DIRECTORY = "NegativeTextures\\"
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

count = 0
class_list = ["NegativeTextures", "ChickyTextures"]
L = 10
positive_list = os.listdir(POSITIVE_TEXTURES_DIRECTORY)
number_positive_files = len(positive_list)
negative_list = os.listdir(NEGATIVE_TEXTURES_DIRECTORY)
number_negative_files = len(negative_list)
label_train = np.zeros((number_positive_files * number_negative_files,1))
print("positive = ", number_positive_files)
print("negative = ", number_negative_files)
features_train = np.zeros((number_positive_files * number_negative_files,4 * L * 3))
for class_id in range(0,2):
    for im_filename in os.listdir(POSITIVE_TEXTURES_DIRECTORY):
        im = cv2.imread(POSITIVE_TEXTURES_DIRECTORY + im_filename)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.resize(im_gray, (50, 50))
        # By default, gray image will have 256 x 256 dimensions
        # We will divide it by 16, that is normalizing all pixels into range 0-15
        # So, it will have 16 x 16 instead
        im_gray = (im_gray / 16).astype(np.uint8)
        # We do not use glcm directly, but the statistical numbers instead
        # range(1, L + 1): 10 offsets
        # [0, np.pi / 4, np.pi / 2]: 3 directions, that are, 0, 90, and 180
        glcm = greycomatrix(im_gray, range(1, L + 1), [0, np.pi / 4, np.pi / 2], 16, symmetric=True, normed=True)
        glcm_props = np.zeros(4 * L * 3) # 4 properties x 3 directions x 10 offsets = 120 features
        glcm_props[0:(L * 3)] = greycoprops(glcm, 'ASM').reshape(1, -1)[0] # Uniformity
        glcm_props[(L * 3):(L * 3 * 2)] = greycoprops(glcm, 'contrast').reshape(1, -1)[0]
        glcm_props[(L * 3 * 2):(L * 3 * 3)] = greycoprops(glcm, 'homogeneity').reshape(1, -1)[0]
        glcm_props[(L * 3 * 3):(L * 3 * 4)] = greycoprops(glcm, 'correlation').reshape(1, -1)[0]
        features_train[count] = glcm_props # 120 features x 45 images --> 45 rows and each row has 120 features
        label_train[count] = 1
        count = count+1
    count = 0
    for im_filename in os.listdir(NEGATIVE_TEXTURES_DIRECTORY):
        im = cv2.imread(NEGATIVE_TEXTURES_DIRECTORY + im_filename)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.resize(im_gray, (50, 50))
        # By default, gray image will have 256 x 256 dimensions
        # We will divide it by 16, that is normalizing all pixels into range 0-15
        # So, it will have 16 x 16 instead
        im_gray = (im_gray / 16).astype(np.uint8)
        # We do not use glcm directly, but the statistical numbers instead
        # range(1, L + 1): 10 offsets
        # [0, np.pi / 4, np.pi / 2]: 3 directions, that are, 0, 90, and 180
        glcm = greycomatrix(im_gray, range(1, L + 1), [0, np.pi / 4, np.pi / 2], 16, symmetric=True, normed=True)
        glcm_props = np.zeros(4 * L * 3) # 4 properties x 3 directions x 10 offsets = 120 features
        glcm_props[0:(L * 3)] = greycoprops(glcm, 'ASM').reshape(1, -1)[0] # Uniformity
        glcm_props[(L * 3):(L * 3 * 2)] = greycoprops(glcm, 'contrast').reshape(1, -1)[0]
        glcm_props[(L * 3 * 2):(L * 3 * 3)] = greycoprops(glcm, 'homogeneity').reshape(1, -1)[0]
        glcm_props[(L * 3 * 3):(L * 3 * 4)] = greycoprops(glcm, 'correlation').reshape(1, -1)[0]
        features_train[count] = glcm_props # 120 features x 45 images --> 45 rows and each row has 120 features
        label_train[count] = 1
        count = count+1

svm = cv2.ml.SVM_create() # Train using SVM
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(features_train.astype(np.float32), cv2.ml.ROW_SAMPLE,label_train.astype(np.int32))

# Read filename list
with open(LIST_FILENAME) as f:
    training_set_list = f.readlines()
training_set_list = [x.strip() for x in training_set_list if x.strip()]

result_output = []
# for IMAGE_FILENAME in os.listdir(CHICKY_SET_DIRECTORY):
#     CHICKY = 1
for IMAGE_FILENAME in training_set_list:
    CHICKY = 0
    outStr = IMAGE_FILENAME + ":"
    foundBox = False
    fileName = re.split('\.', IMAGE_FILENAME)
    CURRENT_FILENAME = fileName[0]
    if CHICKY:
        im_original = cv2.imread(CHICKY_SET_DIRECTORY + IMAGE_FILENAME)
        im_for_gray = cv2.imread(CHICKY_SET_DIRECTORY + IMAGE_FILENAME, 0)
    else:
        im_original = cv2.imread(TRAINING_SET_DIRECTORY + IMAGE_FILENAME)
        im_for_gray = cv2.imread(TRAINING_SET_DIRECTORY + IMAGE_FILENAME, 0)
    im_for_crop = im_original.copy()
    im = adjust_gamma(im_original, 1.9)
    im = cv2.medianBlur(im, 5)
    im = cv2.GaussianBlur(im, (3, 3), 0)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(hsv)
    # s.fill(155)
    # v.fill(255)
    hsv_image = cv2.merge([h, s, v])
    LOWER_YELLOW = np.array([16, 90, 110], dtype=np.uint8)
    UPPER_YELLOW = np.array([95, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
    closing_se = np.ones((35, 35), np.uint8)
    im = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, closing_se)
    closing_se = np.ones((10, 10), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, closing_se)
    # closing_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    # im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, closing_se)


    # opening_se = np.ones((5, 5), np.uint8)
    # im = cv2.morphologyEx(im, cv2.MORPH_OPEN, opening_se)
    # erode_se = np.ones((1, 1), np.uint8)  # structuring element
    # im = cv2.erode(yellow_mask, erode_se, iterations=2)
    im = cv2.medianBlur(im, 5)
    yellow_mask = im.copy()
    # cv2.imshow(CURRENT_FILENAME + ' yellow mask (binary)', yellow_mask)
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
        im_window = im_for_crop[y:(y+h), x:(x+w)]
        im_window = adjust_gamma(im_window, 1.7)
        if im_window.size < 0.011 * im_original.size:
            continue
        hsv_image = cv2.cvtColor(im_window, cv2.COLOR_BGR2HSV_FULL)
        LOWER_RED1 = np.array([170, 90, 70], dtype=np.uint8)
        UPPER_RED1 = np.array([180, 255, 255], dtype=np.uint8)
        LOWER_RED2 = np.array([0, 90, 70], dtype=np.uint8)
        UPPER_RED2 = np.array([15, 255, 255], dtype=np.uint8)
        hsv_image = cv2.medianBlur(hsv_image, 5)
        red_mask1 = cv2.inRange(hsv_image, LOWER_RED1, UPPER_RED1)
        red_mask2 = cv2.inRange(hsv_image, LOWER_RED2, UPPER_RED2)
        red_mask = red_mask1 + red_mask2
        # cv2.imshow(CURRENT_FILENAME + " red_mask (" + str(x) + ", " + str(y) + ", " + str(w) + ", " + str(h) + ")", red_mask)
        LOWER_YELLOW = np.array([29, 120, 110], dtype=np.uint8)
        UPPER_YELLOW = np.array([48, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
        # cv2.imshow(CURRENT_FILENAME + " yellow_mask (" + str(x) + ", " + str(y) + ", " + str(w) + ", " + str(h) + ")", yellow_mask)
        num_red_pixels = cv2.countNonZero(red_mask)
        num_yellow_pixels = cv2.countNonZero(yellow_mask)
        sum_mask = yellow_mask + red_mask
        # if True:
        if ((num_yellow_pixels > num_red_pixels)) \
                and (num_red_pixels < 0.075 * im_window.size) \
                and (num_red_pixels > 0.0006 * im_window.size) \
                and (np.sum(im_window) < 0.69 * np.sum(im_original)):
            box = (x, y, w, h)
            okBoundingBoxes.append(box)
            okNumYellowPixels.append(num_yellow_pixels)
        cv2.rectangle(im_original, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # # for texture training
        # M = moments[numBoxes - 1]
        # if M["m00"] != 0:
        #     cX = int(M["m10"] / M["m00"]) + random.randrange(-10, 10)
        #     cY = int(M["m01"] / M["m00"]) + random.randrange(-10, 10)
        #     cropped_center = im_for_crop[(cY-25):(cY+25),(cX-25):(cX+25)]
        #     cropped_file_name = "UnknownTextures//" + CURRENT_FILENAME + '-' + str(random.randrange(10)) + SAVE_IMAGE_EXTENSION
        #     if cropped_center.size > 0:
        #         cv2.imwrite(cropped_file_name, cropped_center)
        #         # cv2.imshow(cropped_file_name, cropped_center)
    print(CURRENT_FILENAME + ": #boxes = " + str(numBoxes))
    lastBoxes = []
    if okNumYellowPixels:
        idx = 0
        for (x, y, w, h) in okBoundingBoxes:
            im_window = im_for_crop[y:(y+h), x:(x+w)]
            cY = int((y + h) / 2)
            cX = int((x + w) / 2)
            im_gray = im_for_gray[(cY-25):(cY+25),(cX-25):(cX+25)]
            # cv2.imshow("im_gray " + str(x) + str(y), im_gray)
            im_gray = adjust_gamma(im_gray, 1.9)
            # im_gray = cv2.cvtColor(im_gray, cv2.COLOR_BGR2GRAY)
            im_gray = cv2.resize(im_gray, (50, 50))
            im_gray = (im_gray / 16).astype(np.uint8)
            glcm = greycomatrix(im_gray, range(1, L + 1), [0, np.pi / 4, np.pi / 2], 16, symmetric=True, normed=True)
            glcm_props = np.zeros(4 * L * 3)
            glcm_props[0:(L * 3)] = greycoprops(glcm, 'ASM').reshape(1, -1)[0]
            glcm_props[(L * 3):(L * 3 * 2)] = greycoprops(glcm, 'contrast').reshape(1, -1)[0]
            glcm_props[(L * 3 * 2):(L * 3 * 3)] = greycoprops(glcm, 'homogeneity').reshape(1, -1)[0]
            glcm_props[(L * 3 * 3):(L * 3 * 4)] = greycoprops(glcm, 'correlation').reshape(1, -1)[0]
            # Predict the result
            result = svm.predict(glcm_props.reshape(1,-1).astype(np.float32))[1]
            # cv2.imshow(CURRENT_FILENAME+"="+class_list[result[0][0].astype(int)]+"("+str(x)+str(y)+str(w)+str(h)+")",im_gray)
        # okIdx = np.argmax(np.array(okNumYellowPixels))
        # (x, y, w, h) = okBoundingBoxes[okIdx]
            yellowMean = sum(okNumYellowPixels) / float(len(okNumYellowPixels))

            LOWER_ORANGE = np.array([18, 140, 110], dtype=np.uint8)
            UPPER_ORANGE = np.array([23, 255, 255], dtype=np.uint8)
            orange_mask = cv2.inRange(im_window, LOWER_ORANGE, UPPER_ORANGE)
            num_orange_pixels = cv2.countNonZero(orange_mask)
            # cv2.imshow(CURRENT_FILENAME + " orange_mask (" + str(x) + ", " + str(y) + ", " + str(w) + ", " + str(h) + ")", orange_mask)

            if class_list[result[0][0].astype(int)] == "ChickyTextures" \
                and num_orange_pixels < 0.05 * np.sum(im_window):
                # cv2.rectangle(im_original, (x, y), (x + w, y + h), (0, 0, 255), 2)
                foundBox = True
                lastBoxes.append((x,y,w,h))
            # for texture training
            # M = moments[numBoxes - 1]
            # if M["m00"] != 0:
            #     cX = int(M["m10"] / M["m00"]) + random.randrange(-10, 10)
            #     cY = int(M["m01"] / M["m00"]) + random.randrange(-10, 10)
            #     cropped_center = im_for_crop[(cY-25):(cY+25),(cX-25):(cX+25)]
            #     cropped_file_name = "UnknownTextures//" + CURRENT_FILENAME + '-' + str(random.randrange(10)) + SAVE_IMAGE_EXTENSION
            #     if cropped_center.size > 0:
            #         cv2.imwrite(cropped_file_name, cropped_center)
            idx = 0
                    # cv2.imshow(cropped_file_name, cropped_center)

    if foundBox:
        if len(lastBoxes) > 1:
            # boxIdx = random.randint(0, len(lastBoxes)-1)
            boxIdx = 0
        else:
            boxIdx = 0
        (x, y, w, h) = lastBoxes[boxIdx]
        outStr = outStr + str(x) + "," + str(y) + "," + str(w) + "," + str(h)
        cv2.rectangle(im_original, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        outStr = outStr + "none"
    result_output.append(outStr)
    print("outStr = ", outStr)
    # cv2.drawContours(im, contours, -1, (0, 255, 0), 2)
    # cv2.imshow(IMAGE_FILENAME, im_original)
    SaveImage(im_original, 'result')

    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('w'):
    #         break
            # cv2.destroyAllWindows()
            # exit()

with open('570610601.txt', 'w') as f:
    f.write('\n'.join(result_output))
