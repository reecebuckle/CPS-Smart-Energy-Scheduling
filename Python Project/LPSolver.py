import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from itertools import groupby

# Import TestingData (100 abnormal curves). Note this is loaded into Excel CSV to set column headings
guidelineCurves = pd.read_csv("AbnormalCurvesCSV.csv", sep=',')
# Filter out abnormal curves
abnormalCurves = guidelineCurves[guidelineCurves['Label'] == 1]
print(abnormalCurves.head())
# 51 rows, 25 columns
print(abnormalCurves.shape)

# Import the 5 users and 50 tasks dataset
userTasks = pd.read_csv("UsersTasksCSV.csv", sep=',')
print(userTasks.head())

# Iterate through each abnormal guideline curve (~51 of them)
for abnormalCurve, curveRow in abnormalCurves.iterrows():
    print(curveRow[1])

    # Instantiate a Glop solver, naming it SolveStigler.
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create solver objective
    objective = solver.Objective()

    for task, taskRow in userTasks.iterrows():
        # Instantiate a dictionary to store all variables at different hours
        variables = dict()

        # Create the variables
        for num in range(taskRow[1], (taskRow[2] + 1)):
            var = solver.NumVar(0, solver.infinity(), 'x' + str(num))
            variables[num] = var

        varList = []
        for key in variables:
            varList.append(variables[key])
            # Constraint: 0 <= variable for hour <= 1
            # ( variable is between these bounds and less than maximum scheduled energy per hour )
            solver.Add(0 <= variables[key] <= taskRow[3])

        # Constraint: x20 + x21 + x22 + x23 = 3
        # ( sum of variables is equal to energy demand)
        solver.Add(sum(varList) == taskRow[4])

        for key in variables:
            # print("Var_Hour: " + str(key) + ", Unit Cost: " + str(normalCurve[key]))
            objective.SetCoefficient(variables[key], curveRow[key - 1])

    objective.SetMinimization()
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
    else:
        print('The problem does not have an optimal solution.')
