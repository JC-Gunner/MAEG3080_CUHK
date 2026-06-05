# Part 1: Equivalent Code using Scikit-Learn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('ex1data1.txt', sep=',', header=None)
df.columns = ['population', 'profit']

# Prepare features (X) and target (y)
X = df['population'].values.reshape(-1, 1)  # Reshape to 2D array for sklearn
y = df['profit'].values

# Fit linear regression using sklearn
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Get sklearn parameters
sklearn_theta0 = lin_reg.intercept_  # Intercept (theta0)
sklearn_theta1 = lin_reg.coef_[0]    # Slope (theta1)

print("=" * 50)
print("Sklearn Linear Regression Results:")
print("=" * 50)
print(f"Theta0 (intercept): {sklearn_theta0:.4f}")
print(f"Theta1 (slope): {sklearn_theta1:.4f}")

# Make predictions for comparison
X_pred = np.array([[3.5], [7.0]])  # Population in 10,000s
y_pred_sklearn = lin_reg.predict(X_pred)

print(f"\nPredictions for population = 35,000: ${y_pred_sklearn[0] * 10000:.2f}")
print(f"Predictions for population = 70,000: ${y_pred_sklearn[1] * 10000:.2f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X, lin_reg.predict(X), color='red', label='Sklearn Regression Line')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Linear Regression using Sklearn')
plt.legend()
plt.grid(True)
plt.show()

# Compare with Gradient Descent results
print("\n" + "=" * 50)
print("Comparison with Gradient Descent:")
print("=" * 50)

# Run gradient descent for comparison (using code from ex1_linear_regression.py)
def initialize(df):
    m = len(df)
    X_raw = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values.reshape(-1, 1)
    X = np.hstack((np.ones((m, 1)), X_raw))
    theta = np.zeros((2, 1))
    return X, y, theta

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = predictions - y
        gradient = (1/m) * (X.T.dot(error))
        theta = theta - alpha * gradient
    return theta

# Run gradient descent
X_gd, y_gd, theta_gd = initialize(df)
iterations = 1500
alpha = 0.01
theta_final_gd = gradient_descent(X_gd, y_gd, theta_gd, alpha, iterations)

gd_theta0 = theta_final_gd[0][0]
gd_theta1 = theta_final_gd[1][0]

print(f"\nGradient Descent Theta0: {gd_theta0:.4f}")
print(f"Gradient Descent Theta1: {gd_theta1:.4f}")
print(f"\nSklearn Theta0: {sklearn_theta0:.4f}")
print(f"Sklearn Theta1: {sklearn_theta1:.4f}")

# Calculate differences
theta0_diff = abs(gd_theta0 - sklearn_theta0)
theta1_diff = abs(gd_theta1 - sklearn_theta1)

print(f"\nDifferences:")
print(f"Theta0 difference: {theta0_diff:.6f}")
print(f"Theta1 difference: {theta1_diff:.6f}")

# Make predictions with gradient descent for comparison
predict1_gd = np.array([1, 3.5]).dot(theta_final_gd)[0]
predict2_gd = np.array([1, 7.0]).dot(theta_final_gd)[0]

print(f"\nGradient Descent Prediction for 35,000: ${predict1_gd * 10000:.2f}")
print(f"Sklearn Prediction for 35,000: ${y_pred_sklearn[0] * 10000:.2f}")
print(f"Difference: ${abs(predict1_gd - y_pred_sklearn[0]) * 10000:.2f}")

print(f"\nGradient Descent Prediction for 70,000: ${predict2_gd * 10000:.2f}")
print(f"Sklearn Prediction for 70,000: ${y_pred_sklearn[1] * 10000:.2f}")
print(f"Difference: ${abs(predict2_gd - y_pred_sklearn[1]) * 10000:.2f}")

# Analysis of results
print("\n" + "=" * 50)
print("Analysis:")
print("=" * 50)
print("""
Are the results the same? No, they are slightly different.

Why are they different?

1. **Optimization Method**: 
   - Sklearn uses the Normal Equation (closed-form solution) or SVD, which gives the exact optimal solution
   - Gradient Descent is an iterative optimization algorithm that approximates the solution

2. **Convergence**:
   - Gradient Descent may not have fully converged after 1500 iterations
   - The learning rate (0.01) and number of iterations affect how close GD gets to the optimal solution

3. **Numerical Precision**:
   - Different algorithms have different numerical precision characteristics
   - Sklearn's implementation may use more sophisticated numerical methods

4. **Stopping Criteria**:
   - Gradient Descent runs for exactly 1500 iterations (no convergence check)
   - Sklearn's LinearRegression uses analytical solution or optimized numerical methods

The differences are typically small (as seen in the results above), and increasing the number of iterations 
or decreasing the learning rate in Gradient Descent would make the results closer to Sklearn's solution.
""")

# Part 2: Show that with more iterations, GD approaches sklearn solution
print("\n" + "=" * 50)
print("Demonstration: Gradient Descent with More Iterations")
print("=" * 50)

# Run gradient descent with more iterations
theta_gd_more = gradient_descent(X_gd, y_gd, theta_gd, 0.01, 10000)
print(f"GD with 10,000 iterations:")
print(f"Theta0: {theta_gd_more[0][0]:.6f} (Difference from sklearn: {abs(theta_gd_more[0][0] - sklearn_theta0):.6f})")
print(f"Theta1: {theta_gd_more[1][0]:.6f} (Difference from sklearn: {abs(theta_gd_more[1][0] - sklearn_theta1):.6f})")