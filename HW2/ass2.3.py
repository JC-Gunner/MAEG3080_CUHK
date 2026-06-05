import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# Try to load the .mat file
filename = 'ex2data1.mat'
if not os.path.exists(filename):
    print(f"Warning: {filename} not found. Using synthetic data for demonstration.")
    np.random.seed(3080)
    # Generate correlated 2D data (similar to what might be in the file)
    mean = [5, 3]
    cov = [[2, 1.5], [1.5, 2]]
    X = np.random.multivariate_normal(mean, cov, 100)
else:
    data = loadmat(filename)
    # Inspect keys to find the data variable
    # print(data.keys())  # uncomment to see available keys
    # Assume the first non-metadata key contains the data
    for key in data.keys():
        if not key.startswith('__'):
            X = data[key]
            break
    else:
        raise ValueError("No data variable found in .mat file")
    # If you know the key is 'X', use: X = data['X']

# Rest of PCA code as before...

# 1. Normalize data to zero mean
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# 2. Compute covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# 3. SVD to get eigenvectors/eigenvalues
U, S, Vt = np.linalg.svd(cov_matrix)  # U contains eigenvectors as columns
# Principal components are the columns of U (or rows of Vt). We'll take first PC.
pc1 = U[:, 0]  # direction of greatest variance

# 4. Project data onto first principal component (1D)
X_projected_1d = X_centered @ pc1  # shape (m,)

# For visualization, we can also reconstruct the projected points in 2D
X_reconstructed = np.outer(X_projected_1d, pc1) + X_mean

# Plot original and projected data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label='Original data')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.quiver(X_mean[0], X_mean[1], pc1[0]*2, pc1[1]*2, angles='xy', scale_units='xy', scale=1,
           color='red', width=0.02, label='PC1 direction')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original 2D Data')
plt.legend()
plt.axis('equal')

plt.subplot(1, 2, 2)
# Plot original points faintly
plt.scatter(X[:, 0], X[:, 1], alpha=0.2, color='gray')
# Plot projected points (along PC1)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.8, c='red', label='Projected (1D)')
# Draw lines connecting each point to its projection
for i in range(len(X)):
    plt.plot([X[i, 0], X_reconstructed[i, 0]], [X[i, 1], X_reconstructed[i, 1]],
             'k--', lw=0.5, alpha=0.3)
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.quiver(X_mean[0], X_mean[1], pc1[0]*2, pc1[1]*2, angles='xy', scale_units='xy', scale=1,
           color='red', width=0.02)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Projection onto First Principal Component')
plt.legend()
plt.axis('equal')

plt.tight_layout()
plt.savefig('pca_projection.png', dpi=150)
plt.show()