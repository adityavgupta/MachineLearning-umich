# EECS 545 HW3 Q5

# Install scikit-learn package if necessary:
# pip install -U scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

np.random.seed(545)

def readMatrix(filename: str):
    # Use the code below to read files
    with open(filename, 'r') as fd:
        hdr = fd.readline()
        rows, cols = [int(s) for s in fd.readline().strip().split()]
        tokens = fd.readline().strip().split()
        matrix = np.zeros((rows, cols))
        Y = []
        for i, line in enumerate(fd):
            nums = [int(x) for x in line.strip().split()]
            Y.append(nums[0])
            kv = np.array(nums[1:])
            k = np.cumsum(kv[:-1:2])
            v = kv[1::2]
            matrix[i, k] = v
        return matrix, tokens, np.array(Y)


def evaluate(output, label) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    error = float((output != label).sum()) * 1. / len(output)
    print('Error: {:2.4f}%'.format(100 * error))

    return error


def main():
    # Load files
    # Note 1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note 2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    dataMatrix_train, tokenlist, category_train = readMatrix('q5_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q5_data/MATRIX.TEST')

    # Train
    clf = LinearSVC(max_iter=20000)
    clf.fit(dataMatrix_train, category_train)
    pred = clf.predict(dataMatrix_test)
    print("##### 5a ######")
    # test and evaluate
    evaluate(pred, category_test)

    print("\n##### 5b ######")
    #50
    t_list = [50,100,200,400,800,1400]
    errs = []
    for test_set in t_list:
        dmtrain, tklist, ctrain = readMatrix('q5_data/MATRIX.TRAIN.'+str(test_set))
        pred2 = clf.fit(dmtrain, ctrain).predict(dataMatrix_test)
        print("train"+str(test_set)+" error")
        er = evaluate(pred2, category_test)
        errs.append(er*100)
        test_list = abs(clf.decision_function(dmtrain))
        print("Num of support vectors:", sum(i < 1 for i in test_list), "\n")
    
    plt.figure()
    plt.plot(t_list, errs, marker='o')
    plt.xlabel("Training size")
    plt.ylabel("Test Set Error (%)")
    plt.grid(True)
    plt.savefig("q5.png")
    plt.show()
    
if __name__ == '__main__':
    main()
