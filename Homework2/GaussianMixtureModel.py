# Machine Learning 2 Homework 2
print("\nMachine Learning 2 Homework 2")
# By Kutlu Toren & Louis Devèze
print("By Kutlu Toren & Louis Deveze")
# Implementation of a Tri-Dimensional Gaussian Mixture Model
print("Implementation of a Tri-Dimensional Gaussian Mixture Model\n")

# --------------------
# READ ME
# To run the program just enter the command line python GaussianMixtureModel.py import numpy as np 


# ---------------------
# Importing Libraries

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
from mpl_toolkits.mplot3d import axes3d
from sklearn import mixture
import numpy as np
import random
import math

# ---------------------
# Constant

# Number of Samples
I = 100
# Number of cluster
J = 2
# Number of features
N = 3

DISPERSION = 3
RAND_Var = 20
RAND_Mean = 30
rnd = random.Random(10)

# ---------------------
# Function Defines

# Expectation Function
def Expectation(Phi, Mean, Variance, x):

    # Variable to Store numerator of wj(i)
    num = np.zeros((J,I))

    # Foreach J
    for j in range(J):

        #Compute the constant using the determinant of the variance matrix
        const = 1 / ( (2*math.pi)**(N/2) * math.sqrt(np.linalg.det(Variance[j]))) 
        
        # Foreach I
        for i in range(I):

            # Compute the (X(i)-uj) matrix
            distanceMatrix = np.ndarray((N,1))
            for n in range(N):
                distanceMatrix[n] = x[i][n] - Mean[n][j]
            invertSigma = np.linalg.inv(Variance[j])
            transposedDistanceMatrix = np.matrix.transpose(distanceMatrix)
            
            # Final Matrix multiplication result
            value = np.matmul(np.matmul(transposedDistanceMatrix, invertSigma), distanceMatrix)
            num[j][i] = const * math.exp(-0.5*value[0][0]) * Phi[j]

    # Now Compute the denominators
    denominators = np.zeros(I)
    for j in range(J):
        for i in range(I):
            denominators[i] += num[j][i]

    #Finally Compute the wj(i)
    w = np.zeros((J,I))
    for j in range(J):
        for i in range(I):
            w[j][i] = num[j][i] / denominators[i]
            
    return w

# Maximization Function
def Maximization(Phi, Mean, Variance, x, w):


    # Compute the sum of wj(i)
    wSum = np.zeros(J)
    for i in range(I):
        for j in range(J):
            wSum[j] += w[j][i]

    newMean = np.zeros((N,J))
    # First compute new Mean
    for j in range(J):
        for n in range(N):  
            for i in range(I):
                newMean[n][j] += w[j][i] * x[i][n]
            newMean[n][j] = newMean[n][j] / wSum[j]

    # Compute new Phi
    newPhi = wSum / I

    # Compute new Sigmas matrices
    newVariance = np.zeros((J,N,N))
    for j in range(J):
        for i in range(I):
            # Compute the (X(i)-uj) matrix
            distanceMatrix = np.ndarray((N,1))
            for n in range(N):
                distanceMatrix[n] = x[i][n] - newMean[n][j]
            # 
            transposedDistanceMatrix = np.matrix.transpose(distanceMatrix)
            
            newVariance[j] += w[j][i] * np.matmul(distanceMatrix, transposedDistanceMatrix)
        
        newVariance[j] /= wSum[j]

    return newPhi, newMean, newVariance


# ---------------------
# Data Generation

# generate two random Samples
# generate spherical data centered on (20, 20, 20)
shifted_gaussian = np.random.randn(int(3*I/4), 3)*DISPERSION + np.array([22, 22, 22])

# generate 3D Stretched Gaussian Data
B = np.array([[0.9, 0, .3], [.5, .7, 1.0], [.5, .7, 1.0]])
A = np.random.randn(int(I/4), 3)*DISPERSION
stretched_gaussian = np.dot(A, B) + np.array([8, 8, 8])

X_train = np.concatenate((stretched_gaussian, shifted_gaussian), axis=0)


# ---------------------
# Initialization Step

# Phi
Phi = np.zeros(J)
for i in range(J):
    Phi[i] = (1/J)

# Mean
Mean = np.zeros((N,J))
for n in range(N):
    for j in range(J):
        Mean[n][j] = rnd.random() * RAND_Mean


# Variance
Variance = np.zeros((J,N,N))
for j in range(J):
    for n in range(N):
        Variance[j][n][n] = rnd.random() * RAND_Var

# ---------------------
# Training the Model & Found Y*
w = []
for i in range(3):
    w = Expectation(Phi, Mean, Variance, X_train)
    Phi, Mean, Variance = Maximization(Phi, Mean, Variance, X_train, w)

y = []
for i in range(I):
    maxJ = w[0][i]
    maxJindex = 0
    for j in range(J):
        if w[j][i] > maxJ:
            maxJ=w[j][i]
            maxJindex=j
    y.append(maxJindex)

colors=[]
for i in range(I):
    if y[i] == 1:
        colors.append('r')
    else:
        colors.append('g')



# ---------------------
# Plot setup


# Create Plot
fig = plt.figure()
ax = plt.axes(projection="3d")

X = X_train[:, 0]
Y = X_train[:, 1] 
Z = X_train[:, 2]

# Labels
ax.set_xlabel('X Axes') 
ax.set_ylabel('Y Axes') 
ax.set_zlabel('Z Axes')
# Range
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(0, 30)
# Dots
ax.scatter(Mean[0][0],Mean[1][0],Mean[2][0],color='b')
ax.scatter(Mean[0][1],Mean[1][1],Mean[2][1],color='b')
ax.scatter(X,Y,Z,color=colors)
plt.show()