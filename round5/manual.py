import cvxpy as cp
import numpy as np

# Inputs



# sentiments = {
#     'Refrigerators': '+',
#     'Earrings': '++',
#     'Blankets': '---',
#     'Sleds': '---',
#     'Sculptures': '++',
#     'PS6': '+++',
#     'Serum': '----',
#     'Lamps': '+',
#     'Chocolate': '-'
# }
sentiments = {
    'Haystacks': '++',
    'Ranch': '+',
    'Cacti': '----',
    'Solar': '--',
    'Flags': '-',
    'VR': '+++',
    'Coffee': '---',
    'Moonshine': '-',
    'Shirts': '+'
}

returns = {
    '+': 0.05,
    '++': 0.15,
    '+++': 0.35,
    '-': -0.05,
    '--': -0.15,
    '---': -0.4,
    '----': -0.6
}


products = list(sentiments.keys())
rets = np.array([returns[sentiments[p]] for p in products])

# Decision variables (integer percentages)
x = cp.Variable(9, integer=True)

# Objective: Maximise profit = expected return - cost
# Total return = 10_000 * sum(r_i * x_i)
# Total cost   = 90 * sum(x_i^2)
objective = cp.Maximize(10_000 * rets @ x - 120 * cp.sum_squares(x))

# Constraints
constraints = [
    cp.norm1(x) <= 100 
]

# Solve
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.ECOS_BB)

# Results
print("Optimal integer-constrained allocation:")
for i in range(9):
    print(f"{products[i]}: {int(round(x.value[i]))}%")
print(f"\nExpected Profit: {prob.value:.2f} SeaShells")
print(f"Total % used: {sum(abs(round(x.value[i])) for i in range(9))}%")