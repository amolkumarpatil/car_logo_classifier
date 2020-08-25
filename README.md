## Car Logo Classification
This tool use Histogram of Oriented Gradient (HoG) apporach along with Machine Learning to solve the problem statement.
Below are the classes used for classification :-
* Buick
* Chery
* Citroen
* Honda
* Hyundai
* Lexus
* Mazda
* Peugeot
* Toyota
* VW

## Description
Images used for classification are 70 x x 70 mixed bit depth images i.e. single channel/ RGB images with 10 different car logos. The dataset was picked from kaggle (link below). Images apart from being various color schemes, are at different angles with noise.

Tool classifies a given image by attributing a predicted brand label: 

## Model Matrics
Classification model logistic regression got below scores:- 
* Accuracy - 0.95
* Recall   - 0.94 
* F1 Score - 0.95
* Precision -0.95
## Training
To train classification model run below command:
`python main.py -train <directory_to_training_images> -classifier <model_name>`

Below models are supported by tool currently:
* Linear SVM
* Logistic Regression
* KNN
## Testing
 To test classification model run below command:
`python main.py -model <model_path>
-test <directory_to_test_images>`

## Requirements
* sklearn
* skimage
* imutils
* opencv-python
* pickle
* joblib
* numpy