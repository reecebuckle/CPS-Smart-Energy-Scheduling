from sklearn import svm
from sklearn.model_selection import train_test_split
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

# # grab all hour columns
X = training_set[
    ['Hour 0', 'Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7', 'Hour 8', 'Hour 9', 'Hour 10',
     'Hour 11', 'Hour 12', 'Hour 13', 'Hour 14', 'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18', 'Hour 19', 'Hour 20',
     'Hour 21', 'Hour 22', 'Hour 23']]

# Separate Training Curves
normal_curves = training_set[training_set['Label'] == 0]
abnormal_curves = training_set[training_set['Label'] == 1]

# Iterate through 5000 normal guide line curves,
# Record features such as PAR, min/max unit cost and frequency of low occurrences (less than 3)
normal_PARS = []
normal_unit_costs = []
normal_low_occurrences = 0
normal_hours_0to8 = []
normal_hours_9to23 = []

for curve_ID, row in normal_curves.iterrows():
    unit_costs = []
    for x in range(24):
        unit_costs.append(row[x])
        normal_unit_costs.append(row[x])
        if row[x] < 3:
            normal_low_occurrences += 1

    # Calculate peak to average ratio for each curve
    peak = max(unit_costs)
    average = (sum(unit_costs) / len(unit_costs))
    PAR = peak / average
    normal_PARS.append(PAR)

    # Collect feature data about the first 8 hours of a day
    unit_costs.clear()
    for x in range(9):
        unit_costs.append(row[x])

    average = (sum (unit_costs) / len(unit_costs))
    normal_hours_0to8.append(average)

    # Collect feature data about the last 14 hours of a day
    unit_costs.clear()
    for x in range(9, 24):
        unit_costs.append(row[x])

    average = (sum(unit_costs) / len(unit_costs))
    normal_hours_9to23.append(average)


# Iterate through 5000 abnormal guide line curves,
# Record features such as PAR, min/max unit cost and frequency of low occurrences (less than 3)
abnormal_PARS = []
abnormal_unit_costs = []
abnormal_low_occurrences = 0
abnormal_hours_0to8 = []
abnormal_hours_9to23 = []

for curve_ID, row in abnormal_curves.iterrows():
    unit_costs = []
    for x in range(24):
        unit_costs.append(row[x])
        abnormal_unit_costs.append(row[x])
        if row[x] < 3:
            abnormal_low_occurrences += 1

    # Calculate peak to average ratio for each curve
    peak = max(unit_costs)
    average = (sum(unit_costs) / len(unit_costs))
    PAR = peak / average
    abnormal_PARS.append(PAR)

    # Collect feature data about the first 8 hours of a day
    unit_costs.clear()
    for x in range(9):
        unit_costs.append(row[x])

    average = (sum(unit_costs) / len(unit_costs))
    abnormal_hours_0to8.append(average)

    # Collect feature data about the last 14 hours of a day
    unit_costs.clear()
    for x in range(9, 24):
        unit_costs.append(row[x])

    average = (sum(unit_costs) / len(unit_costs))
    abnormal_hours_9to23.append(average)

# Calculate the average of 5000 abnormal PARs
average_abnormal_PAR = (sum(abnormal_PARS) / len(abnormal_PARS))
abnormal_average_cost_hours9to23 = (sum(abnormal_hours_9to23) / len(abnormal_hours_9to23))
# Print out some stats about the abnormal features
print("\nFeatures of ABNORMAL guideline curves")
print("Average PAR: ", average_abnormal_PAR)                    # 1.345852220801892
print("Max unit cost: ", max(abnormal_unit_costs))              # 6.96944450925151
print("Min unit cost: ", min(abnormal_unit_costs))              # 2.66303905615289
print("Number of low occurrences: ", abnormal_low_occurrences)  # 2352 low occurrences
print("Average unit cost (hours 0-8 inclusive)", (sum(abnormal_hours_0to8) / len(abnormal_hours_0to8)))
print("Average unit cost (hours 9-23 inclusive)", (sum(abnormal_hours_9to23) / len(abnormal_hours_9to23)))

# Calculate the average of 5000 normal PARs
average_normal_PAR = (sum(normal_PARS) / len(normal_PARS))
normal_average_cost_hours9to23 = (sum(normal_hours_9to23) / len(normal_hours_9to23))
# Print out some stats about the normal features
print("\nFeatures of NORMAL guideline curves")
print("Average PAR: ", average_normal_PAR)                    # 1.3644166405368692
print("Max unit cost: ", max(normal_unit_costs))              # 6.96921266337082
print("Min unit cost: ", min(normal_unit_costs))              # 2.66312815366513
print("Number of low occurrences: ", normal_low_occurrences)  # 3774 low occurrences
print("Average unit cost (hours 0-8 inclusive)", (sum(normal_hours_0to8) / len(normal_hours_0to8)))
print("Average unit cost (hours 9-23 inclusive)", normal_average_cost_hours9to23)

# Establish a threshold as a halfway point between these two ratios, as the normal PAR is higher
threshold = average_normal_PAR - ((average_normal_PAR - average_abnormal_PAR) / 2)
# Threshold = 1.3551344306693807
print("\nThreshold PAR (difference between two PARs): ", threshold)

# Iterate through all curves in the training set, and classify them based on the PAR threshold
# if > threshold = classify as normal, if < threshold, classify as abnormal

success = 0
# Iterate through all 10000 guideline curves
for curve_ID, row in training_set.iterrows():
    unit_costs = []
    for x in range(24):
        unit_costs.append(row[x])

    # Calculate peak to average ratio for each curve
    peak = max(unit_costs)
    average = (sum(unit_costs) / len(unit_costs))
    PAR = peak / average

    # Collect feature data about the last 14 hours of a day
    unit_costs.clear()
    for x in range(9, 24):
        unit_costs.append(row[x])

    average = (sum(unit_costs) / len(unit_costs))

    # Classify based on result
    # If PAR is greater than threshold, this is typically normal BUT
    if PAR > threshold:

        # only classify normal if average cost between hours 9 - 23 is less than the abnormal average
        if average < abnormal_average_cost_hours9to23:
            if row[24] == 0:
                success += 1

        # Else classify it as abnormal!
        else:
            if row[24] == 1:
                success += 1

    # Else if PAR is less than threshold, this is typically abnormal BUT
    elif PAR < threshold:

        # only classify abnormal if average cost between hours 9-23 is greater than normal average
        if average > normal_average_cost_hours9to23:
            if row[24] == 1:
                success += 1

        # Else classify normal again
        else:
            if row[24] == 0:
                success += 1

# Use total successes to evaluate accuracy of this approach
print("\nSuccess rate: ", success)
print("Accuracy: ", success / 10000 * 100, "%")
print("Error Rate: ", 100 - (success / 10000 * 100), "%")
