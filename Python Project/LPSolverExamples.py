import pandas as pd
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt

# Example Normal curve from spread sheet
normal_curve = [4.246522377, 3.640027796, 3.480502639, 3.245460995, 3.162915992, 3.597667495,
                3.905355954, 4.078340246, 5.374797426, 4.944699124, 5.438100083, 3.909231366,
                6.200666726, 4.482141894, 5.410801558, 6.149170969, 5.8837687, 6.329263208,
                6.469511152, 5.58349762, 5.558922379, 5.255354797, 5.568480613, 5.441475567]

# Example Abnormal curve from spread sheet
abnormal_curve = [4.246522377, 3.640027796, 3.480502639, 3.245460995, 3.162915992, 3.597667495,
                  3.905355954, 4.078340246, 5.374797426, 4.944699124, 5.438100083, 3.909231366,
                  6.200666726, 4.482141894, 5.410801558, 6.149170969, 5.8837687, 6.329263208,
                  3.40001, 5.58349762, 5.558922379, 5.255354797, 5.568480613, 5.441475567]

# Import the 5 users and 50 tasks dataset
print("---------- User Tasks Dataset -----------")
user_tasks = pd.read_csv("UsersTasksCSV.csv", sep=',')
print(user_tasks.head())
print(user_tasks.shape)


# Instantiate a Glop solver to solve problem
solver = pywraplp.Solver.CreateSolver('GLOP')

# Create solver objective
objective = solver.Objective()

# Instantiate a dictionary to store all variables (FOR ALL ROWS). Form: { Key: x23 : Value: 23 }
all_variables = dict()

for task, row in user_tasks.iterrows():
    # Temp dictionary to store variables on each row. CLEARED ON EACH ROW. Form: { Key 23 : Value x23 }
    variables = dict()

    # Create the variables from the ready to deadline (inclusive)
    for num in range(row[1], (row[2] + 1)):
        var = solver.NumVar(0, solver.infinity(), 'x' + str(num))
        variables[num] = var
        all_variables[var] = num

    # Temp list to hold all variables in this row (constraint 2)
    row_variables = []

    # Constraint: 1 <= variable <= 1 (less than maximum scheduled energy per hour)
    for key in variables:
        solver.Add(0 <= variables[key] <= row[3])
        row_variables.append(variables[key])

    # Constraint 2: x20 + x21 + x22 + x23 = 3 (sum of variables in row is equal to energy demand)
    solver.Add(sum(row_variables) == row[4])

    # Set objective (min cost function) coefficients
    # cost: hour 0 cost * (sum of x0 vars) + ... + hour 23 cost * (sum of x23 vars)
    for key in variables:
        objective.SetCoefficient(variables[key], abnormal_curve[key])

# Solve for for the min cost
status = solver.Solve()

# Min Cost List
cost_per_hour = []

# If there is an optimal minimised solution
if status == pywraplp.Solver.OPTIMAL:
    print('Total MIN cost for all 5 users:', solver.Objective().Value())

    # Iterate through all variables to sum all variable solutions at each hour
    # This is the minimal cost solution (for all users) for the barchart
    for hour in range(0, 24):
        min_cost = 0
        for key, value in all_variables.items():
            if value == hour:
                min_cost += key.solution_value()

        cost_per_hour.append(min_cost)
        # Print sum of value
        print("Sum of all values in hour", hour, ": ", min_cost)

    # Print peak to average (PAR) ratio (peak energy divided by average energy)
    print("PAR: ", (max(cost_per_hour) / (sum(cost_per_hour) / len(cost_per_hour))))

else:
    print('The problem does not have an optimal solution')

# Produce bar char using a list of all the min cost per hour
hour = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
        '20', '21', '22', '23']

plt.bar(hour, cost_per_hour)
plt.ylabel("Total Unit Cost (Consumption)")
plt.xlabel("Time Slot (Hour)")
plt.title("Abnormal Curve from Example Spreadsheet \n Min cost solution for all 5 users: "
          + str(round(solver.Objective().Value(), 2)))

# plt.show()
plt.savefig('AbnormalExampleChart.png', dpi=300)
