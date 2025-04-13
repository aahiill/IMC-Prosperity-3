# Fixed container data: (multiplier, inhabitants)
containers = [
    (10, 1),
    (80, 6),
    (31, 2),
    (37, 3),
    (17, 1),
    (50, 4),
    (90, 10),
    (20, 2),
    (73, 4),
    (89, 8)
]

def calculate_value(multiplier, inhabitants, percentage):
    """Calculate expected value from container"""
    return multiplier * (10000 / (inhabitants + percentage))

# Get competition expectations from user
print("Enter expected percentage of people who will choose each container (0-100):")
percentages = []

for i, (multiplier, inhabitants) in enumerate(containers):
    while True:
        try:
            percentage = float(input(f"Container {i+1} ({multiplier}x, {inhabitants} inh.): "))
            if 0 <= percentage <= 100:
                percentages.append(percentage)
                break
            else:
                print("Please enter a number between 0 and 100")
        except ValueError:
            print("Please enter a valid number")

# Calculate and display results
print("\n----- RESULTS -----")
print("Container\tMultiplier\tInhabitants\tPercentage\tValue")
print("-" * 70)

for i, ((multiplier, inhabitants), percentage) in enumerate(zip(containers, percentages)):
    value = calculate_value(multiplier, inhabitants, percentage)
    worth_it = "✓" if value > 50000 else " "
    print(f"{i+1}\t\t{multiplier}\t\t{inhabitants}\t\t{percentage}%\t\t{value:.0f} {worth_it}")

# Find best container
best_idx = max(range(len(containers)), key=lambda i: calculate_value(containers[i][0], containers[i][1], percentages[i]))
best_mult, best_inh = containers[best_idx]
best_perc = percentages[best_idx]
best_value = calculate_value(best_mult, best_inh, best_perc)

print("\nBest container:", end=" ")
print(f"Container {best_idx+1} ({best_mult}x, {best_inh} inh.) with value {best_value:.0f} SeaShells")