from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from mlxtend.evaluate import bias_variance_decomp
from sklearn import metrics
import pandas as pd
import os

headings = ['Hour 0', 'Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7', 'Hour 8', 'Hour 9',
            'Hour 10', 'Hour 11', 'Hour 12', 'Hour 13', 'Hour 14', 'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18',
            'Hour 19', 'Hour 20', 'Hour 21', 'Hour 22', 'Hour 23', 'Label']

# Importing the training/testing txt dataset using pandas, with custom headings
training_set = pd.read_csv("TrainingData.txt", sep=',', names=headings)
headings.pop()
testing_set = pd.read_csv("TestingData.txt", sep=',', names=headings)

# Display datasets
print("--------  Displaying training data set  --------")
print(training_set.head())
print(training_set.tail())

print("\n--------  Displaying testing data set  --------")
print(testing_set.head())
print(testing_set.tail())

# grab the 5000 0's or 1's columns to train on
Y = training_set['Label']

# grab all hour columns
X = training_set[
    ['Hour 0', 'Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7', 'Hour 8', 'Hour 9', 'Hour 10',
     'Hour 11', 'Hour 12', 'Hour 13', 'Hour 14', 'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18', 'Hour 19', 'Hour 20',
     'Hour 21', 'Hour 22', 'Hour 23']]

# Uncomment these two lines to calculate an estimated mse, bias and variance
# data = training_set.values
# X, Y = data[:, :-1], data[:, -1]

# Split the training set 70% testing / 30% training, and randomise data selected
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

# Create a class linear svm classifier
classifier = svm.SVC(kernel='linear', C=1, random_state=42)

# Train the classifier
classifier.fit(X_train, Y_train)

# Perform cross validation to avoid bias
scores = cross_val_score(classifier, X, Y, cv=5)
print("\nCross Validation Scores: ", scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Uncomment these lines to estimate bias and variance
# mse, bias, var = bias_variance_decomp(classifier, X_train, Y_train, X_test, Y_test, loss='mse', num_rounds=200, random_seed=1)
# # summarize results
# print('MSE: %.3f' % mse)
# print('Bias: %.3f' % bias)
# print('Variance: %.3f' % var)

# Grab predicted results
Y_predicted = classifier.predict(X_test)

# Print how accurate the model is
print("\nAccuracy: ", metrics.accuracy_score(Y_test, Y_predicted))
# Print some analysis metrics
print("\nconfusion matrix: \n", confusion_matrix(Y_test, Y_predicted))
print("\nreport: \n", classification_report(Y_test, Y_predicted))

# Use trained classifier on the testing set (100 unlabelled curves)
prediction_results = classifier.predict(testing_set)

# Print the prediction results
print("\nPredicted Labels: ", prediction_results)

# Output a TestingResults.txt file
os.remove("TestingResults.txt")
testing_set["Label"] = prediction_results
testing_set.to_csv('TestingResults.txt', header=None, index=None, sep=',', mode='a')

# Display classified results datasets
testing_results = pd.read_csv("TestingResults.txt", sep=',', header=None)
print("\n --------  Displaying training data set  --------")
print(testing_results.head())
print(testing_results.tail())
print(testing_results.shape)
