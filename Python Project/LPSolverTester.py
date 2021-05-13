import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from itertools import groupby

# Example Normal curve from spread sheet
normalCurve = [4.246522377, 3.640027796, 3.480502639, 3.245460995, 3.162915992, 3.597667495,
               3.905355954, 4.078340246, 5.374797426, 4.944699124, 5.438100083, 3.909231366,
               6.200666726, 4.482141894, 5.410801558, 6.149170969, 5.8837687, 6.329263208,
               6.469511152, 5.58349762, 5.558922379, 5.255354797, 5.568480613, 5.441475567]

# Example Abnormal curve from spread sheet
abnormalCurve = [4.246522377, 3.640027796, 3.480502639, 3.245460995, 3.162915992, 3.597667495,
                 3.905355954, 4.078340246, 5.374797426, 4.944699124, 5.438100083, 3.909231366,
                 6.200666726, 4.482141894, 5.410801558, 6.149170969, 5.8837687, 6.329263208,
                 3.40001, 5.58349762, 5.558922379, 5.255354797, 5.568480613, 5.441475567]


# Import the 5 users and 50 tasks dataset
userTasks = pd.read_csv("UsersTasksCSV.csv", sep=',')
print(userTasks.head())

# Instantiate a Glop solver, naming it SolveStigler.
solver = pywraplp.Solver.CreateSolver('GLOP')

# Create solver objective
objective = solver.Objective()

for task, row in userTasks.iterrows():
    # Instantiate a dictionary to store all variables at different hours
    # { Key : Value}
    variables = dict()

    # Create the variables
    for num in range(row[1], (row[2] + 1)):
        var = solver.NumVar(0, solver.infinity(), 'x' + str(num))
        variables[num] = var

    varList = []
    for key in variables:
        varList.append(variables[key])
        # Constraint: 0 <= variable for hour <= 1
        # ( variable is between these bounds and less than maximum scheduled energy per hour )
        solver.Add(0 <= variables[key] <= row[3])

    # Constraint: x20 + x21 + x22 + x23 = 3
    # ( sum of variables is equal to energy demand)
    solver.Add(sum(varList) == row[4])

    for key in variables:
        # print("Var_Hour: " + str(key) + ", Unit Cost: " + str(normalCurve[key]))
        objective.SetCoefficient(variables[key], abnormalCurve[key - 1])

objective.SetMinimization()
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Hour 0 =', x0.solution_value())
    print('Hour 0 =', x0.solution_value())
    print('Hour 0 =', x0.solution_value())
    print('Hour 0 =', x0.solution_value())
    print('Hour 0 =', x0.solution_value())
    print('Hour 0 =', x0.solution_value())


else:
    print('The problem does not have an optimal solution.')
