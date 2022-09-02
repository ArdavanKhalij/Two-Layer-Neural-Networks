################################################################################
# Libraries
import numpy as np
import matplotlib.pyplot as plt
################################################################################
# Data
λ = 0.75
ε = 20
NumberOfUpdates = 100
NumberOfLayers = 2
NumberOfUnits = [4, 2]
Input = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
ERRORS = []
################################################################################
# Weights
Weights = []
for i in range(NumberOfLayers - 1):
    Weights.append(np.random.random((NumberOfUnits[i] + 1, NumberOfUnits[i + 1])) - 0.5)
Weights.append(np.random.random((NumberOfUnits[-1] + 1, NumberOfUnits[0])) - 0.5)
################################################################################
# Equation 2 in the paper
def Logistic(x):
    return 1 / (1 + np.exp(-x))
################################################################################
# The loop of training
for i in range(0, NumberOfUpdates):
    X = []
    for n in NumberOfUnits:
        X.append(np.ones((Input.shape[0], 2, n + 1)))
    # Step 1: first pass
    X[0][:, 0, :-1] = Input
    for i in range(NumberOfLayers - 1):
        X[i + 1][:, 0, :-1] = Logistic(X[i][:, 0, :] @ Weights[i])
    # Step 2: second pass
    X[0][:, 1, :-1] = λ * X[0][:, 0, :-1] + (1 - λ) * Logistic(X[-1][:, 0, :] @ Weights[-1])
    for i in range(1, NumberOfLayers):
        X[i][:, 1, :-1] = λ * X[i][:, 0, :-1] + (1 - λ) * Logistic(X[i - 1][:, 1, :] @ Weights[i - 1])
    # Updating the weights
    for i in range(len(Weights) - 1):
        Weights[i] += ε * X[i][:, 1, :].T @ (X[i + 1][:, 0, :-1] - X[i + 1][:, 1, :-1])
    Weights[-1] += ε * X[-1][:, 0, :].T @ (X[0][:, 0, :-1] - X[0][:, 1, :-1])
    ERRORS.append(0.5 * ((Input - X[0][:, 1, :-1]) ** 2).sum())
################################################################################
# Plot for illustration of one time run with 100 updates
fig = plt.figure()
ax = fig.subplots(1, 1)
ax.plot(ERRORS)
ax.set_title('Training Error')
ax.set_ylabel('Mean Squared Error')
ax.set_xlabel('Number Of Generations')
plt.show()
################################################################################