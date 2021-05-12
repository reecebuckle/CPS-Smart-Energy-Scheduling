import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from itertools import groupby

# Unit cost for each hour of an abnormal curve
curveOne = [4.038200717, 3.874220935, 3.120742561, 3.261642723, 2.990716865, 3.789114849, 3.935849303, 4.39182354,
            5.356574916, 5.274407608, 5.439402898, 3.823221124, 6.003448749, 4.263088319, 5.8222691, 6.206443924,
            5.631746969, 6.631983059, 6.593440703, 5.643768061, 5.930986061, 5.421772574, 5.150518763, 5.126661159]

# Import the 5 users and 50 tasks dataset
LPDataset = pd.read_csv("UsersCSV.csv", sep=',')
print(LPDataset.head())

# Instantiate a Glop solver, naming it SolveStigler.
solver = pywraplp.Solver.CreateSolver('GLOP')

# Create solver objective
objective = solver.Objective()

# Instantiate a dictionary to store all variables at different hours
variables = dict()

# Create the variables
for num in range(20, 23 + 1):
    var = solver.NumVar(0, solver.infinity(), 'x' + str(num))
    variables[num] = var

varList = []
for key in variables:
    varList.append(variables[key])

# Add the constraint to the solver
solver.Add(sum(varList) == 1)

print(varList)

for key in variables:
    objective.SetCoefficient(variables[key], curveOne[key])


status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())


else:
    print('The problem does not have an optimal solution.')
