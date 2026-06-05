import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def initialize(df):
   
    m = len(df)
    
    # Reshape features and target to ensure they are 2D arrays (m x 1)
    # Column 0 is population (x), Column 1 is profit (y) [cite: 15]
    X_raw = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values.reshape(-1, 1)
    
    # Add a column of ones to X to absorb the bias term theta_0 [cite: 29]
    # np.hstack concatenates arrays horizontally
    X = np.hstack((np.ones((m, 1)), X_raw))
    
    # Initialize theta as zeros (2 x 1 vector)
    theta = np.zeros((2, 1))
    
    return X, y, theta

def compute_cost_one_variable(X, y, theta):
   
    m = len(y)
    
    # Calculate predictions: f_theta(x) = X * theta [cite: 41]
    predictions = X.dot(theta)
    
    # Calculate squared errors
    sq_errors = (predictions - y) ** 2
    
    # Calculate Cost J [cite: 40]
    J = (1 / (2 * m)) * np.sum(sq_errors)
    
    return J

def gradient_descent(X, y, theta, alpha, num_iters, df):
   
    m = len(y)
    J_history = [] # To store cost at every iteration
    
    for i in range(num_iters):
        # 1. Calculate prediction
        predictions = X.dot(theta)
        
        # 2. Calculate error (prediction - actual)
        error = predictions - y
        
        # 3. Calculate gradient
        # Gradient formula derived from partial derivative of cost function
        gradient = (1/m) * (X.T.dot(error))
        
        # 4. Update theta
        theta = theta - alpha * gradient
        
        # Save the cost J in every iteration for visualization later
        J_history.append(compute_cost_one_variable(X, y, theta))
        
        # Visualization: Plot data with decision boundary every 500 iterations 
        if i % 500 == 0:
            print(f"Iteration {i}: Cost {J_history[-1]:.4f}")
            plt.figure(figsize=(8, 6))
            # Plot the raw training data
            sns.scatterplot(x='population', y='profit', data=df, label='Training Data')
            
            # Plot the current regression line (hypothesis)
            # x values from dataframe, y values are our current predictions
            plt.plot(df['population'], X.dot(theta), color='red', label=f'Hypothesis (Iter {i})')
            
            plt.xlabel('Population of City in 10,000s')
            plt.ylabel('Profit in $10,000s')
            plt.title(f'Regression Line at Iteration {i}')
            plt.legend()
            plt.show()

    return theta, J_history

def main():
    # Load Data [cite: 14]
    try:
        df = pd.read_csv('ex1data1.txt', sep=',', header=None)
        df.columns = ['population', 'profit']
    except FileNotFoundError:
        print("Error: 'ex1data1.txt' not found. Make sure the file is in the same directory.")
        return

    # Initial Visualization [cite: 21]
    ax = sns.scatterplot(x='population', y='profit', data=df)
    ax.set(xlabel='Population of City in 10,000s', ylabel='Profit in $10,000s',
           title='Scatter plot of training data')
    plt.show()

    ## Start Implementation
    X, y, theta = initialize(df)
    iterations = 1500  # [cite: 44]
    alpha = 0.01       # [cite: 44]
    
    print("Initial Cost (theta=0):", compute_cost_one_variable(X, y, theta))
    
    # Run Gradient Descent
    # Note: Updated to unpack both theta and cost_history
    theta_final, cost_history = gradient_descent(X, y, theta, alpha, iterations, df)

    print("\nFinal theta:", theta_final.flatten())
    print("Final cost:", cost_history[-1])
    
    # Plot cost history (Verification step)
    plt.figure()
    plt.plot(range(iterations), cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.grid(True)
    plt.show()
    
    # Final prediction examples
    # Prediction for population = 35,000 (input 3.5) and 70,000 (input 7.0)
    # We must manually add the bias term (1) to the input vector: [1, 3.5]
    predict1 = np.array([1, 3.5]).dot(theta_final)
    predict2 = np.array([1, 7.0]).dot(theta_final)
    
    print(f"\nPrediction for population = 35,000: ${predict1[0] * 10000:.2f}")
    print(f"Prediction for population = 70,000: ${predict2[0] * 10000:.2f}")
    
    # Final Plot with learned parameters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='population', y='profit', data=df, label='Training Data')
    plt.plot(df['population'], X.dot(theta_final), color='green', label='Final Hypothesis')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Final Linear Regression Fit')
    plt.legend()
    plt.show()

if __name__ =='__main__':
    main()