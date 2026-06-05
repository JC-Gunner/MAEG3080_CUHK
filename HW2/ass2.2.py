import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

np.random.seed(3080)

# Load Iris dataset, use first two features
iris = load_iris()
X = iris.data[:, :2]  # sepal length and sepal width

# K-means implementation from scratch
def kmeans(X, K, max_iters=100, tol=1e-4):
    # Randomly initialize centroids
    n_samples, n_features = X.shape
    indices = np.random.choice(n_samples, K, replace=False)
    centroids = X[indices]
    
    for i in range(max_iters):
        # Assign points to nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids, rtol=tol):
            break
        centroids = new_centroids
    
    return centroids, labels

K = 3
centroids, labels = kmeans(X, K)

# Plot results
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for k in range(K):
    plt.scatter(X[labels == k, 0], X[labels == k, 1],
                c=colors[k], label=f'Cluster {k+1}', alpha=0.6, edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200,
            label='Centroids', edgecolors='white')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('K-Means Clustering on Iris (first two features)')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_iris.png', dpi=150)
plt.show()