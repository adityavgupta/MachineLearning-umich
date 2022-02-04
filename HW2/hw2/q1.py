from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

def sigmoid (s):
    return 1/(1+np.exp(-s))

def weightsFromNewtonRaphson(X, Y, nIters):
    w = np.zeros((X.shape[1],1))
    for i in range(nIters):
        s = sigmoid(X@w)
        d = -s*(1-s)
        D = np.diag(d.T[0])
        H = X.T@(D@X)
        w = w - np.linalg.inv(H)@(X.T@(Y-s))
    print(w)
    return w
def main():
    # We format the data matrix so that each row is the feature for one sample.
    # The number of rows is the number of data samples.
    # The number of columns is the dimension of one data sample.
    X = np.load('q1x.npy')
    N = X.shape[0]
    Y = np.load('q1y.npy').reshape((N,1))
    # To consider intercept term, we append a column vector with all entries=1.
    # Then the coefficient correpsonding to this column is an intercept term.
    X = np.concatenate((np.ones((N, 1)), X), axis=1)
    w = weightsFromNewtonRaphson(X,Y,100)
    xd = np.array([np.min(X), np.max(X)])
    b = -w[0,0]/w[2,0]
    m = -w[1,0]/w[2,0]
    yd = m*xd + b
    _0_idx = (Y[:,0]==0).nonzero()[0]
    _1_idx = (Y[:,0]==1).nonzero()[0]
    plt.figure(figsize=(8,6))
    plt.plot(xd, yd, label="Decision boundary")
    plt.scatter(X[:,1][_0_idx], X[:,2][_0_idx], color="blue", label="0", marker="^")
    plt.scatter(X[:,1][_1_idx], X[:,2][_1_idx], color="red", label="1")
    plt.xlabel("x1");plt.ylabel("x2");
    plt.legend(loc="best")
    plt.title("True lables and decision boundary")
    plt.grid(True)
    plt.savefig('q1-c.png')
    plt.show()
if __name__ == "__main__":
    main()
        
