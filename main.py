# import the necessary packages
import argparse
import train
import test

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-train", "--training_path", required=False, help="Path to training dataset")
ap.add_argument("-test", "--test_path", required=False, help="Path to test dataset")
ap.add_argument("-model", "--model_path", required=False, help="Path saved_model")
ap.add_argument("-classifier", "--classifier", required=False, help="use proper string : svm, lr, knn")
args = vars(ap.parse_args())

if args['training_path'] !="" and args['classifier'] :
    # train.train(r"I:\workspace\projects\opencv\car_logo_detector\TrainingData\small_train")
    if args['classifier'] == 'svm' or args['classifier'] == 'lr' or args['classifier'] == 'knn':
        train.train(args['training_path'], args['classifier'])
    else :
        print("Please select proper classifier")
elif args['test_path'] !="" and args['model_path'] !="":
    test.test(args['model_path'],args['test_path'])