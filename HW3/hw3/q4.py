# EECS 545 HW3 Q4

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(545)

# Instruction: use these hyperparameters for both (b) and (d)
eta = 0.5
C = 5
iterNums = [5, 50, 100, 1000, 5000, 6000]


def svm_train_bgd(matrix: np.ndarray, label: np.ndarray, nIter: int):
    # Implement your algorithm and return state (e.g., learned model)
    N, D = matrix.shape

    ##################################
    # TODO: Implement your code here #
    ##################################
    w = np.zeros((D,1))
    b = 0
    
    for j in range(1,nIter+1):
        indic = np.where(label * (matrix @ w + b) < 1, 1, 0)
        w_grad = w - C*(matrix.T@(indic*label)) # Nx1
        b_grad = np.sum(-C*(indic*label), axis=0)[0]
        alpha = eta/(1+j*eta)
        w = w - alpha*w_grad
        b = b - 0.01*alpha*b_grad
    return {'w':w, 'b':b}


def svm_train_sgd(matrix: np.ndarray, label: np.ndarray, nIter: int):
    # Implement your algorithm and return state (e.g., learned model)
    
    N, D = matrix.shape

    ##################################
    # TODO: Implement your code here #
    ##################################
    w = np.zeros((D,1))
    b = 0

    for j in range(1,nIter+1):
        alpha = eta/(1+j*eta)
        for i in range(N):
            h = (matrix[i]@w)+b
            w_grad = (1/N)*w - C*(label[i]*h < 1)*(label[i]*(matrix[i].reshape(-1,1)))
            b_grad = -C*(label[i]*h < 1)*label[i]
            w = w - alpha*w_grad
            b = b - 0.01*alpha*b_grad
    state = {"w":w, "b":b}
    
    return state


def svm_test(matrix: np.ndarray, state):
    # Classify each test data as +1 or -1
    output = np.ones( (matrix.shape[0], 1) )

    ##################################
    # TODO: Implement your code here #
    ##################################
    y_hat = (matrix @ state['w']) + state['b']
    output = np.where(y_hat > 0, 1, -1)
    return output


def evaluate(output: np.ndarray, label: np.ndarray, nIter: int, state) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    accuracy = (label * output > 0).sum() * 1. / len(output)
    
    print('Iter {:4d}:\naccuracy = {:2.4f}%'.format(nIter, 100 * accuracy))
    print("w: ", state['w'])
    print("b: ", state['b'], "\n")

    return accuracy


def load_data():
    # Note1: label is {-1, +1}
    # Note2: data matrix shape  = [Ndata, 4]
    # Note3: label matrix shape = [Ndata, 1]

    # Load data
    q4_data = np.load('q4_data/q4_data.npy', allow_pickle=True).item()

    train_x = q4_data['q4x_train']
    train_y = q4_data['q4y_train']
    test_x = q4_data['q4x_test']
    test_y = q4_data['q4y_test']
    return train_x, train_y, test_x, test_y


def run_bgd(train_x, train_y, test_x, test_y):
    '''(c) Implement SVM using **batch gradient descent**.
    For each of the nIter's, print out the following:

    *   Parameter w
    *   Parameter b
    *   Test accuracy (%)
    '''
   
    for nIter in iterNums:
        # Train
        state = svm_train_bgd(train_x, train_y, nIter)

        # TODO: Test and evluate
        prediction = svm_test(test_x, state)
        evaluate(prediction, test_y, nIter, state)


def run_sgd(train_x, train_y, test_x, test_y):
    '''(c) Implement SVM using **stocahstic gradient descent**.
    For each of the nIter's, print out the following:

    *   Parameter w
    *   Parameter b
    *   Test accuracy (%)

    [Note: Use the same hyperparameters as (b)]
    [Note: If you implement it correctly, the running time will be ~15 sec]
    '''
    for nIter in iterNums:
        # Train
        state = svm_train_sgd(train_x, train_y, nIter)

        # TODO: Test and evluate
        prediction = svm_test(test_x, state)
        evaluate(prediction, test_y, nIter, state)


def main():
    train_x, train_y, test_x, test_y = load_data()
    print("###### SVM Batch Gradient Descent ######")
    run_bgd(train_x, train_y, test_x, test_y)
    print("###### SVM Stochastic Gradient Descent ######")
    run_sgd(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
