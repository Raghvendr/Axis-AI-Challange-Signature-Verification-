import numpy as np
np.random.seed(1)
import random
random.seed(1)
import cv2

def preprocess_image(raw_image, display=False):
    # Converting the image chnnel to black and hite
    bw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    bw_image = 255 - bw_image
    # thresholding the image
    threshold, threshold_image = cv2.threshold(bw_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # finding Contours
    im2, contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(cv2.findNonZero(threshold_image))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    image1 = cv2.drawContours(threshold_image.copy(), [box], 0, (120, 120, 120), 2)
    (x, y) = np.where(image1 == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    # Extracting text from Contours
    out = threshold_image[topx:bottomx+1, topy:bottomy+1]
    resized = cv2.resize(out, (150,80), interpolation = cv2.INTER_AREA) 
    return resized

def extract_image(image_path):
    img = cv2.imread(image_path)
    h, w, channels = img.shape
    img = img[60:h,0:w-60] 
    preprocessed_image =preprocess_image(img)
    flattenimage =  preprocessed_image.flatten()
    image = np.array(flattenimage)
    img_data = image.astype('float32')
    img_data /= 255
    return img_data

