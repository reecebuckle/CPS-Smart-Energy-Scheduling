# CPS-Smart-Energy-Scheduling

In order to run the 5 python scripts in 'Python Project', please navigate to 'Python Project', 
install pip and then run the requirements.txt to install the relevent libraries used in this project

This can be done with: 

python -m pip install -U pip

pip install - r requirements.txt


An explanation of the 5 scripts:

Classifiers.py demonstrates an accuracy of all available classifiers in the scikit-learn library.

SVMClassifier.py demonstrates using an SVM classifier, with a linear kernel, to predict TestingData.txt (and classify these results). 

StatisticalMethods.py demonstrates a custom made algorithm for classifying the data (72% accuracy).

All 3 of these classification scripts will automatically load 'TestingData.txt' and 'TrainingData.txt' from the project folder.


LPExample.py uses the 2 example predictive guideline curves to solve for a minimum cost solution, and output two barcharts.

LPSolver.py will load the 'TestingResults.txt' from the project folder, in order to output a minimum cost solution for all classified guideline curves with an abnormal label.
