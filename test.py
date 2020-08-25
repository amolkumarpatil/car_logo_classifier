from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import imutils
import cv2
import os
import numpy as np
import pickle
import joblib

def test(model_path, image_path):
    # load the model from disk
    model = joblib.load(model_path)
    # loop over the test dataset
    for (i, imagePath) in enumerate(paths.list_images(image_path)):
        # load the test image, convert it to grayscale, and resize it to
        # the canonical size
        print(imagePath)
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logo = cv2.resize(gray, (200, 100))

        # extract Histogram of Oriented Gradients from the test image and
        # predict the make of the car
        (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
        pred = model.predict(H.reshape(1, -1))[0]

        # visualize the HOG image
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        # cv2.imshow("HOG Image #{}".format(i + 1), hogImage)

        # draw the prediction on the test image and display it
        cv2.putText(image, pred.title(), (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 1)
        cv2.imshow("Test Image #{}".format(i + 1), image)
        cv2.waitKey(0)
        filename = str(i)+".jpg"
        cv2.imwrite(filename,image)

if __name__ == "__main__":
    # model_path = r"I:\workspace\projects\opencv\car_logo_detector\svm.sav"
    model_path = input("Enter model path : ")
    image_path = input("Enter image path : ")
    test(model_path, image_path)