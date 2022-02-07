import numpy as np
import matplotlib.pyplot as plt

def gradAsc(X, Y, alpha=0.0005):
    w = np.random.rand(3, 4)
    w[-1,:] = 0

    eps = 10**-8
    loss = 100
    err_new = E(X, Y, w)
    count = 0
    while (loss > eps):
        err_old = err_new
        w[:2,:] = w[:2,:] + alpha * gradient(X, Y, w[:2,:])
        err_new = E(X, Y, w)
        loss = np.abs(err_new - err_old)**2 / np.abs(err_old)**2
        # print("Loss: ", loss)
        count += 1
    print("Num iterations: ", count)
    return w

def gradient(X, Y, w):
    grad = w.copy()
    for i in range(2):
        indicator = np.zeros((X.shape[0], 1))
        indicator[Y == float(i+1)] = 1

        grad[i, :] = np.sum(X * (indicator - prob(X, Y, w, i)), axis = 0)
    return grad

def prob(X, Y, w, m):
    num = np.exp(w[m,:]@X.T).reshape(-1, 1)
    denom = 1 + np.sum(np.exp(w@X.T), axis=0, keepdims=True).T
    return num / denom

def E(X, Y, w):
    err = 0
    for i in range(3):
        indicator = np.zeros((X.shape[0], 1))
        indicator[Y == float(i+1)] = 1
        temp = np.log(prob(X, Y, w, i) ** indicator)
        err += np.sum(temp)
    print(indicator)
    return err

def accuracyChecker(w, X, Y):
    p1and2 = (np.exp(w[:2,:] @ X.T) / (1 + np.sum(np.exp(w @ X.T), axis=0, keepdims=True))).T
    p3 = (1 - np.sum(p1and2, axis=1)).reshape(-1, 1)
    p = np.c_[p1and2, p3]
    print("probability: ", p)

    pred = (np.argmax(p, axis=1) + 1).reshape(-1, 1)
    correct = np.zeros(Y.shape)
    correct[pred == Y] = 1

    return np.sum(correct) / Y.shape[0]


def main():
# Load data
    q2_data = np.load('q2_data.npz')

    q2x_train = q2_data["q2x_train"]
    q2y_train = q2_data["q2y_train"]
    q2x_test = q2_data["q2x_test"]
    q2y_test = q2_data["q2y_test"]

    # print("q2x_train shape: ", q2x_train.shape)
    # print("q2y_train shape: ", q2y_train.shape)

    weights = gradAsc(q2x_train, q2y_train)
    # print("Weights: ", weights)

    prob = accuracyChecker(weights, q2x_test, q2y_test)
    print("Testing accuracy: ", prob)

if __name__ == "__main__":
    main()
