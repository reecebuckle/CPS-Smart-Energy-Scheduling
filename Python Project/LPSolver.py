import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

# Unit cost for each hour of an abnormal curve
curveOne = [4.038200717, 3.874220935, 3.120742561, 3.261642723, 2.990716865, 3.789114849, 3.935849303, 4.39182354,
            5.356574916, 5.274407608, 5.439402898, 3.823221124, 6.003448749, 4.263088319, 5.8222691, 6.206443924,
            5.631746969, 6.631983059, 6.593440703, 5.643768061, 5.930986061, 5.421772574, 5.150518763, 5.126661159]

# Import the 5 users and 50 tasks dataset
LPDataset = pd.read_csv("UsersCSV.csv", sep=',')
print(LPDataset.head())

# Instantiate a Glop solver, naming it SolveStigler.
solver = pywraplp.Solver.CreateSolver('GLOP')

# Create the variables

x20 = solver.NumVar(0, solver.infinity(), 'x20')
x21 = solver.NumVar(0, solver.infinity(), 'x21')
x22 = solver.NumVar(0, solver.infinity(), 'x22')
x23 = solver.NumVar(0, solver.infinity(), 'x23')

print('Number of variables =', solver.NumVariables())

# Create the constraints
# Constraint x20 + x21 + x22 + x23 = 1 (all variables = energy demand)
solver.Add(x20 + x21 + x22 + x23 == 1)

print('Number of constraints =', solver.NumConstraints())


# Objective cost function: (h20 * x20) + (h21 * x21) + (h22 * x22) + (h23 * x23)
solver.Minimize(curveOne[20]*x20 + curveOne[21]*x21 + curveOne[22]*x22 + curveOne[23]*x23)

# Solve the system
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('x20 =', x20.solution_value())
    print('x21 =', x21.solution_value())
    print('x22 =', x22.solution_value())
    print('x23 =', x23.solution_value())
else:
    print('The problem does not have an optimal solution.')
