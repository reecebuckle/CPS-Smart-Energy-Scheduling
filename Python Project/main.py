# sklearn library
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Pandas library
import pandas

# Importing the datasets using pandas
trainingSet = pandas.read_csv("TrainingDataCSV.csv", sep=',')
testingSet = pandas.read_csv("TestingDataCSV.csv", sep=',')

# Display datasets
print("Displaying training set table")
print(trainingSet.head())
print(trainingSet.shape)

print("Displaying testing set table")
print(testingSet.head())
print(testingSet.shape)

# grab the 5000 0's or 1's columns to train on
Y = trainingSet['Normal']

# grab all hour columns
X = trainingSet[
    ['Hour 0', 'Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7', 'Hour 8', 'Hour 9', 'Hour 10',
     'Hour 11', 'Hour 12', 'Hour 13', 'Hour 14', 'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18', 'Hour 19', 'Hour 20',
     'Hour 21', 'Hour 22', 'Hour 23']]

# Create svm classifier
svmClassifier = svm.SVC()

# Split the training set 70% testing / 30% training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Train the classifier
svmClassifier.fit(X_train, Y_train)

# Grab predicted results
Y_predicted = svmClassifier.predict(X_test)

# Print how accurate the model is
print("Accuracy:", metrics.accuracy_score(Y_test, Y_predicted))


