import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Set random seed for reproducibility
np.random.seed(3080)

# 1. Generate XOR dataset
X = np.random.randn(300, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)  # True for points in quadrants 2 and 4

# 2. Train SVM with different kernels
kernels = ['linear', 'rbf', 'poly']
params = [
    {'kernel': 'linear', 'C': 1.0},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    {'kernel': 'poly', 'C': 1.0, 'degree': 3, 'gamma': 'scale', 'coef0': 1}
]

# Create a mesh for decision boundary plotting
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, param in enumerate(params):
    # Train SVM
    clf = SVC(**param)
    clf.fit(X, y)
    
    # Plot decision boundary and margins
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax = axes[i]
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, colors=['#FFAAAA', '#AAAAFF', '#AAFFAA'])
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    
    # Plot data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', s=20)
    
    # Plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               facecolors='none', edgecolors='red', linewidths=1.5, label='Support Vectors')
    
    ax.set_title(f"Kernel: {param['kernel']}")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.legend()

plt.tight_layout()
plt.savefig('svm_xor.png', dpi=150)
plt.show()