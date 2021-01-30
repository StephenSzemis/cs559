# Author: Stephen Szemis
# Pledge: I pledge my honor that I have abided by the Stevens honor system. - Stephen Szemis
# Date: December, 9, 2020

import numpy as np

np.set_printoptions(precision=3)

data = np.array([[5.9, 3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], [5.0, 3.0],
                 [4.9, 3.1], [6.7, 3.1], [5.1, 3.8], [6.0, 3.0]])

cluster_colors = ["RED", "GREEN", "BLUE"]
cluster_centers = np.array([[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]])

def update_centers():
    k = np.shape(cluster_centers)[0]
    distances = np.zeros((len(data), len(cluster_centers)))
    # Get sets for each cluster
    for i, x in enumerate(data):
        for j, y in enumerate(cluster_centers):
            distances[i][j] = np.sqrt(np.sum((x - y) ** 2))
    cluster_assignments = np.argmin(distances, axis=1)

    # Recompute Cluster means
    for i in range(k):
        t = data[cluster_assignments == i]
        cluster_centers[i] = np.sum(t, axis=0) / len(t)

for i in range(4):
    update_centers()
    print("iteration : ", i+1)
    print("center of RED is", cluster_centers[0, :])
    print("center of GREEN is", cluster_centers[1, :])
    print("center of BLUE is", cluster_centers[2, :])
    print("--------------------------------------------------")