import numpy as np
import imutils
import cv2
import os
IMAGE_SIZE = (256, 256)

def preprocess_image(img):
    #Make image black and white, add blur, apply threshold on values, remove areas of noise
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grey, (3, 3), 0)
    threshold = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)[1]
    erode = cv2.erode(threshold, None, iterations=2)
    dilate = cv2.dilate(erode, None, iterations=2)
    #find contours to crop image
    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) == 0:
        return img
    cnt = max(contours, key=cv2.contourArea)
    #find extremes of contour
    left = tuple(cnt[cnt[:, :, 0].argmin()][0])
    right = tuple(cnt[cnt[:, :, 0].argmax()][0])
    top = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])
    #crop image
    cropped = grey[top[1]:bottom[1], left[0]:right[0]].copy()
    #improve contrast
    thresh = cv2.threshold(cropped, 15, 255, cv2.THRESH_TOZERO)[1]
    equalize = cv2.equalizeHist(thresh)
    normalize = cv2.normalize(equalize, None, 0, 255, cv2.NORM_MINMAX)
    return normalize

if __name__ == "__main__":
    training = "images/Training"
    testing = "images/Testing"

    for folder in os.listdir(training):
        path = os.path.join(training, folder)
        clean = "dataset/Training/" + folder
        if not os.path.exists(clean):
            os.makedirs(clean)
        for i in os.listdir(path):
            img = cv2.imread(os.path.join(path, i))
            new_img = preprocess_image(img)
            new_img = cv2.resize(new_img, IMAGE_SIZE)
            np.save(os.path.join(clean, i.replace(".jpg", ".npy")), new_img)
            cv2.imwrite(os.path.join(clean, i), new_img)
    for folder in os.listdir(testing):
        path = os.path.join(testing, folder)
        clean = "dataset/Testing/" + folder
        if not os.path.exists(clean):
            os.makedirs(clean)
        for i in os.listdir(path):
            img = cv2.imread(os.path.join(path, i))
            new_img = preprocess_image(img)
            new_img = cv2.resize(new_img, IMAGE_SIZE)
            np.save(os.path.join(clean, i.replace(".jpg", ".npy")), new_img)
            cv2.imwrite(os.path.join(clean, i), new_img)