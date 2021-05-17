import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt

headings = ['Hour 0', 'Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7', 'Hour 8', 'Hour 9',
            'Hour 10', 'Hour 11', 'Hour 12', 'Hour 13', 'Hour 14', 'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18',
            'Hour 19', 'Hour 20', 'Hour 21', 'Hour 22', 'Hour 23', 'Label']

# Import TestingData (100 abnormal curves). Note this is loaded into Excel CSV to set column headings
guideline_curves = pd.read_csv("TestingResults.txt", sep=',', names=headings)

# Filter out abnormal curves
abnormal_curves = guideline_curves[guideline_curves['Label'] == 1]
normal_curves = guideline_curves[guideline_curves['Label'] == 0]

print("\n ----------- Abnormal Curves Data --------------")
print(abnormal_curves.head())

print("\n ----------- Normal Curves Data --------------")
print(normal_curves.head())

# Import the 5 users and 50 tasks dataset
user_tasks = pd.read_csv("UsersTasksCSV.csv", sep=',')
print("\n ----------- User Tasks Data --------------")
print(user_tasks.head())

# Iterate through each abnormal guideline curve (~51 of them)
for curve_ID, curve_row in abnormal_curves.iterrows():
    # Instantiate a Glop solver, naming it SolveStigler.
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create solver objective
    objective = solver.Objective()

    # Instantiate a dictionary to store all variables (FOR ALL ROWS). Form: { Key: x23 : Value: 23 }
    all_variables = dict()

    for task_ID, task_row in user_tasks.iterrows():
        # Temp dictionary to store variables on each row. CLEARED ON EACH ROW. Form: { Key 23 : Value x23 }
        variables = dict()

        # Create the variables from the ready to deadline (inclusive)
        for num in range(task_row[1], (task_row[2] + 1)):
            var = solver.NumVar(0, solver.infinity(), 'x' + str(num))
            variables[num] = var
            all_variables[var] = num

        # Temp list to hold all variables in this row (constraint 2)
        row_variables = []

        # Constraint: 1 <= variable <= 1 (less than maximum scheduled energy per hour)
        for key in variables:
            solver.Add(0 <= variables[key] <= task_row[3])
            row_variables.append(variables[key])

        # Constraint 2: x20 + x21 + x22 + x23 = 3 (sum of variables in row is equal to energy demand)
        solver.Add(sum(row_variables) == task_row[4])

        # Set objective (min cost function) coefficients
        # cost: hour 0 cost * (sum of x0 vars) + ... + hour 23 cost * (sum of x23 vars)
        for key in variables:
            objective.SetCoefficient(variables[key], curve_row[key])

    # Solve for for the min cost
    status = solver.Solve()

    # Min Cost List
    cost_per_hour = []

    # Print out min cost solution for all 5 users for that curve
    print("Curve number", curve_ID+1, '. Total MIN cost:', solver.Objective().Value())

    # Iterate through all variables to sum all variable solutions at each hour
    # This is the minimal cost solution (for all users) for the barchart
    for hour in range(0, 24):
        min_cost = 0
        for key, value in all_variables.items():
            if value == hour:
                min_cost += key.solution_value()

        cost_per_hour.append(min_cost)

    # Hour variables used in bar chart
    hour = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '22', '23']


    plt.bar(hour, cost_per_hour)
    plt.ylabel("Total Unit Cost / Consumption")
    plt.xlabel("Time Slot (Hour)")
    plt.yticks(np.arange(min(cost_per_hour), max(cost_per_hour) + 1, 1.0))
    plt.title("Abnormal Curve #" + str(curve_ID+1) + "\n Min cost solution for all 5 users: "
              + str(round(solver.Objective().Value(), 2)))
    plt.savefig("AbnormalCharts/AbnormalCurve"+str(curve_ID+1)+".png")

    # Clear plot data for next iteration
    plt.clf()
