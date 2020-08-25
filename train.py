from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import model_selection
from skimage import exposure
from skimage import feature
from imutils import paths
import imutils
import cv2
import os
import numpy as np
import pickle


def get_maker(imagePath):
    dir_name = os.path.dirname(imagePath)
    maker = os.path.basename(dir_name)
    return maker


def select_classifier(classifier):
    if classifier == 'svm':
        model = SVC()
    elif classifier == 'lr':
        model = LogisticRegression()
    elif classifier == 'knn':
        model = KNeighborsClassifier(n_neighbors=1)
    return model


def train(train_path, classifier):
# loop over the image paths in the training set
# initialize the data matrix and labels
    print("[INFO] extracting features...")
    data = []
    labels = []
    
    for imagePath in paths.list_images(train_path):
        try:
            # print(imagePath)
            # extract the make of the car
            # make = imagePath.split("/")[-2]
            maker = get_maker(imagePath)
        
            # load the image, convert it to grayscale, and detect edges
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = imutils.auto_canny(gray)
            # cv2.imshow("edged",edged)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # find contours in the edge map, keeping only the largest one which
            # is presmumed to be the car logo
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
        
            # extract the logo of the car and resize it to a canonical width
            # and height
            (x, y, w, h) = cv2.boundingRect(c)
            logo = gray[y:y + h, x:x + w]
            logo = cv2.resize(logo, (200, 100))
        
            # extract Histogram of Oriented Gradients from the logo
            H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)

            # update the data and labels
            data.append(H[0])
            labels.append(maker)
        except Exception as e:
            print("Error in processing file", imagePath)
    # print(data[0][0])
    # "train" the nearest neighbors classifier
    # data1 = np.asarray(data)
    # labels = np.asarray(labels)
    # print(list(data).shape)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, labels, test_size=0.2, random_state=42)
    print("[INFO] training classifier...")
    model = select_classifier(classifier)
    # model = KNeighborsClassifier(n_neighbors=1)
    # model = LogisticRegression()
    # model = SVC()
    model.fit(X_train, Y_train)
    print("trained successfully")
    # save the model to disk
    model_name = classifier +'.sav'
    pickle.dump(model, open(model_name, 'wb'))
    result = model.score(X_test, Y_test)
    print(result)
    prediction = model.predict(X_test)
    print("")
    print("Classification Report:")
    print(classification_report(Y_test,prediction))
    return data, labels
