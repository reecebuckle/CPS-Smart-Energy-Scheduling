import pandas as pd
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt

# Import TestingData (100 abnormal curves). Note this is loaded into Excel CSV to set column headings
guidelineCurves = pd.read_csv("AbnormalCurvesCSV.csv", sep=',')
# Filter out abnormal curves
abnormalCurves = guidelineCurves[guidelineCurves['Label'] == 1]
# 51 rows, 25 columns
print(abnormalCurves.head())

# Import the 5 users and 50 tasks dataset
userTasks = pd.read_csv("UsersTasksCSV.csv", sep=',')
print(userTasks.head())

# Iterate through each abnormal guideline curve (~51 of them)
for abnormalCurve, curveRow in abnormalCurves.iterrows():
    # Instantiate a Glop solver, naming it SolveStigler.
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create solver objective
    objective = solver.Objective()

    # Instantiate a dictionary to store all variables (FOR ALL ROWS). Form: { Key: x23 : Value: 23 }
    allVariables = dict()

    for task, taskRow in userTasks.iterrows():
        # Temp dictionary to store variables on each row. CLEARED ON EACH ROW. Form: { Key 23 : Value x23 }
        variables = dict()

        # Create the variables from the ready to deadline (inclusive)
        for num in range(taskRow[1], (taskRow[2] + 1)):
            var = solver.NumVar(0, solver.infinity(), 'x' + str(num))
            variables[num] = var
            allVariables[var] = num

        # Temp list to hold all variables in this row (constraint 2)
        rowVariables = []

        # Constraint: 1 <= variable <= 1 (less than maximum scheduled energy per hour)
        for key in variables:
            solver.Add(0 <= variables[key] <= taskRow[3])
            rowVariables.append(variables[key])

        # Constraint 2: x20 + x21 + x22 + x23 = 3 (sum of variables in row is equal to energy demand)
        solver.Add(sum(rowVariables) == taskRow[4])

        # Set objective (min cost function) coefficients
        # cost: hour 0 cost * (sum of x0 vars) + ... + hour 23 cost * (sum of x23 vars)
        for key in variables:
            objective.SetCoefficient(variables[key], curveRow[key])

    # Solve for for the min cost
    status = solver.Solve()

    # Min Cost List
    costPerHour = []

    # Print out min cost solution for all 5 users for that curve
    print("Curve number", abnormalCurve, '. Total MIN cost:', solver.Objective().Value())

    # Iterate through all variables to sum all variable solutions at each hour
    # This is the minimal cost solution (for all users) for the barchart
    for hour in range(0, 24):
        minCost = 0
        for key, value in allVariables.items():
            if value == hour:
                minCost += key.solution_value()

        costPerHour.append(minCost)

    # Hour variables used in bar chart
    hour = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '22', '23']

    plt.bar(hour, costPerHour)
    plt.ylabel("Unit Cost Per Hour")
    plt.xlabel("Hour Starting From")
    plt.title("Abnormal Curve #" + str(abnormalCurve) + "\n Min cost solution for all 5 users: "
              + str(round(solver.Objective().Value(), 2)))
    plt.savefig("AbnormalCharts/AbnormalCurve"+str(abnormalCurve)+".png")

    # Clear plot data for next iteration
    plt.clf()
