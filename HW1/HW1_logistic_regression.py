import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as opt

def initialize(df):
    # TODO: start your code here
    m = len(df)
    X = np.hstack([np.ones((m, 1)), 
                   df['exam_score_1'].values.reshape(-1, 1),
                   df['exam_score_2'].values.reshape(-1, 1)])
    y = df['label'].values.reshape(-1, 1)
    initial_theta = np.zeros((3, 1))
    return X, y, initial_theta

def sigmoid(z):
    # TODO: start your code here
    return 1 / (1 + np.exp(-z))

def compute_cost_function(theta, X, y):
    # TODO: start your code here
    m = len(y)
    h = sigmoid(X @ theta)
    
    # Avoid log(0) errors
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    
    J = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    # Compute gradient
    grad = (1/m) * (X.T @ (h - y))
    
    return J, grad

def optimize_theta(X, y, initial_theta):
    # TODO: start your code here
    # Flatten theta for scipy
    initial_theta = initial_theta.flatten()
    
    # Define the optimization function
    def cost_function(theta):
        return compute_cost_function(theta, X, y)[0]
    
    def gradient_function(theta):
        return compute_cost_function(theta, X, y)[1].flatten()
    
    # Use scipy's minimize function
    result = opt.minimize(fun=cost_function,
                         x0=initial_theta,
                         method='BFGS',
                         jac=gradient_function,
                         options={'maxiter': 400, 'disp': False})
    
    opt_theta = result.x.reshape(-1, 1)
    cost = result.fun
    
    return opt_theta, cost

def predict(X, theta):
    # TODO: start your code here
    probabilities = sigmoid(X @ theta)
    y_pred = (probabilities >= 0.5).astype(int)
    return y_pred

def plot_decision_boundary(X, y, theta):
    # Plot data points
    plt.figure(figsize=(7,5))
    admitted = y.flatten() == 1
    not_admitted = y.flatten() == 0
    
    plt.scatter(X[admitted, 1], X[admitted, 2], marker='+', color='blue', label='Admitted', s=80)
    plt.scatter(X[not_admitted, 1], X[not_admitted, 2], marker='o', color='red', label='Not admitted', s=80)
    
    # Plot decision boundary
    plot_x = np.array([np.min(X[:,1]) - 2, np.max(X[:,1]) + 2])
    plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x, plot_y, color='green', label='Decision Boundary')
    
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def main():
    df = pd.read_csv('ex1data2.txt', sep=',', header=None)
    df.columns = ['exam_score_1', 'exam_score_2', 'label']
    
    plt.figure(figsize=(7,5))
    ax = sns.scatterplot(x='exam_score_1', y='exam_score_2', hue='label', data=df, style='label', s=80)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:], ['Not admitted', 'Admitted'])
    plt.title('Scatter plot of training data')
    plt.show()

    X, y, theta = initialize(df)
    
    # Test sigmoid function
    print("Sigmoid of 0:", sigmoid(0))
    print("Sigmoid of large positive:", sigmoid(10))
    print("Sigmoid of large negative:", sigmoid(-10))
    
    cost, grad = compute_cost_function(theta, X, y)
    print(f"\nInitial cost: {cost}")
    print(f"Initial gradient shape: {grad.shape}")
    
    opt_theta, opt_cost = optimize_theta(X, y, theta)
    print(f"\nOptimized cost: {opt_cost}")
    print(f"Optimized theta: {opt_theta.flatten()}")
    
    y_pred = predict(X, opt_theta)
    accuracy = np.mean(y_pred == y) * 100
    print(f"\nTrain accuracy: {accuracy:.2f}%")
    
    # Plot decision boundary
    plot_decision_boundary(X, y, opt_theta)

if __name__ == '__main__':
    main()