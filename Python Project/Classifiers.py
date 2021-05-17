import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# This class finds the best accuracy from a list of imported classifiers (sklearn library)

headings = ['Hour 0', 'Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7', 'Hour 8', 'Hour 9',
            'Hour 10', 'Hour 11', 'Hour 12', 'Hour 13', 'Hour 14', 'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18',
            'Hour 19', 'Hour 20', 'Hour 21', 'Hour 22', 'Hour 23', 'Label']

# Importing the training/testing txt dataset using pandas, with custom headings
training_set = pd.read_csv("TrainingData.txt", sep=',', names=headings)
headings.pop()
testing_set = pd.read_csv("TestingData.txt", sep=',', names=headings)

# grab the 5000 0's or 1's columns to train on
Y = training_set['Label']
#
# grab all hour columns
X = training_set[
    ['Hour 0', 'Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7', 'Hour 8', 'Hour 9', 'Hour 10',
     'Hour 11', 'Hour 12', 'Hour 13', 'Hour 14', 'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18', 'Hour 19', 'Hour 20',
     'Hour 21', 'Hour 22', 'Hour 23']]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# Split the training set 70% testing / 30% training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

index = 0
for classifier in classifiers:
    print("\n Testing accuracy of: ", names[index])
    index += 1

    # Train the classifier
    classifier.fit(X_train, Y_train)

    # Grab predicted results
    Y_predicted = classifier.predict(X_test)

    # Print how accurate the model is
    print("\n Accuracy: ", metrics.accuracy_score(Y_test, Y_predicted))
    print("---------------------------------------------")

