import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
def softmaxRegGA(X, Y, alpha, nIters, N,M,K):
    w = np.zeros((M,K))
    indic = np.array([[1 if Y[i] == m else 0 for m in range(K-1)]
                     for i in range(N)])

    for iter in range(nIters):
        denom = (1+np.sum(np.exp(X@w)[:, 0:K-1], axis=1)).reshape(-1,1)
        p = np.exp(X@w[:,0:K-1])/denom
        grad = X.T @ (indic - p)
        w[:,0:K-1] += alpha*grad
    return w

def main():
    #  Load data
    q2_data = np.load('q2_data.npz')
    q2_data_files = q2_data.files

    q2x_train = np.array(q2_data[q2_data_files[0]])
    q2x_test = np.array(q2_data[q2_data_files[1]])
    q2y_train = np.array(q2_data[q2_data_files[2]])
    q2y_test = np.array(q2_data[q2_data_files[3]])
    N = q2x_train.shape[0]
    M = q2x_train.shape[1]
    K = int(max(q2y_train)[0])+1
    
    trained_weights = softmaxRegGA(q2x_train, q2y_train, 0.0005, 400, N, M, K)
    denom = np.sum(np.exp(q2x_test@trained_weights), axis=1).reshape(-1,1)
    p = np.exp(q2x_test@trained_weights)/denom
    pred = np.argmax(p, axis=1)
    
    print("Accuracy:", np.sum(pred == q2y_test.T[0])/q2y_test.shape[0])
    # clf = LogisticRegression()
    # clf.fit(q2x_train, q2y_train.flatten().tolist())
    # print(q2y_test.flatten().tolist())
    # print(clf.score(q2x_test, q2y_test.flatten().tolist()))
    
if __name__=="__main__":
    main()