import numpy as np
import matplotlib.pyplot as plt

def readMatrix(file):
    # Use the code below to read files
    fd = open(file, 'r')
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

def nb_train(matrix, category):
    # Implement your algorithm and return 
    
    M = matrix.shape[1]
    
    ############################
    # Implement your code here #
    ############################
    n_j_spam = category @ matrix
    n_j_ns = (1-category)@matrix

    phi = np.sum(category)/len(category)
    mu_js = (n_j_spam+1)/(np.sum(n_j_spam)+M) 
    mu_jns = (n_j_ns+1)/(np.sum(n_j_ns)+M)
    print(mu_jns)
    return phi, mu_js.reshape(-1,1), mu_jns.reshape(-1,1)

def nb_test(matrix, phi, mu_js, mu_jns):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    
    ############################
    # Implement your code here #
    ############################
    log_P_s = np.log(phi) + matrix@np.log(mu_js)
    log_P_ns = np.log(1-phi)+ matrix@np.log(mu_jns)
    output = (log_P_s > log_P_ns).flatten()
    return output

def evaluate(output, label):
    # Use the code below to obtain the accuracy of your algorithm
    error = (output != label).sum() * 1. / len(output)
    print('Error: {:2.4f}%'.format(100*error))
    return error
   

def top5spam(mu_js, mu_jns ,tokens):
    diffs = (np.log(mu_js)-np.log(mu_jns)).flatten()
    return np.flip(tokens[np.argsort(diffs)[-5:]])

def multiData():
    dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')
    dataMatrix_train50, tokenlist, category_train50 = readMatrix('q4_data/MATRIX.TRAIN.50')
     # Train
    phi, mu_js, mu_jns = nb_train(dataMatrix_train50, category_train50)
    # Test and evluate
    prediction = nb_test(dataMatrix_test, phi, mu_js, mu_jns)
    print("Training size: 50")
    e1 = 100*evaluate(prediction, category_test)

    dataMatrix_train100, tokenlist, category_train100 = readMatrix('q4_data/MATRIX.TRAIN.100')
     # Train
    phi, mu_js, mu_jns = nb_train(dataMatrix_train100, category_train100)
    # Test and evluate
    prediction = nb_test(dataMatrix_test, phi, mu_js, mu_jns)
    print("Training size: 100")
    e2 = 100*evaluate(prediction, category_test)

    dataMatrix_train200, tokenlist, category_train200 = readMatrix('q4_data/MATRIX.TRAIN.200')
     # Train
    phi, mu_js, mu_jns = nb_train(dataMatrix_train200, category_train200)
    # Test and evluate
    prediction = nb_test(dataMatrix_test, phi, mu_js, mu_jns)
    print("Training size: 200")
    e3 = 100*evaluate(prediction, category_test)

    dataMatrix_train400, tokenlist, category_train400 = readMatrix('q4_data/MATRIX.TRAIN.400')
     # Train
    phi, mu_js, mu_jns = nb_train(dataMatrix_train400, category_train400)
    # Test and evluate
    prediction = nb_test(dataMatrix_test, phi, mu_js, mu_jns)
    print("Training size: 400")
    e4 = 100*evaluate(prediction, category_test)

    dataMatrix_train800, tokenlist, category_train800 = readMatrix('q4_data/MATRIX.TRAIN.800')
     # Train
    phi, mu_js, mu_jns = nb_train(dataMatrix_train800, category_train800)
    # Test and evluate
    prediction = nb_test(dataMatrix_test, phi, mu_js, mu_jns)
    print("Training size: 800")
    e5 = 100*evaluate(prediction, category_test)

    dataMatrix_train1400, tokenlist, category_train1400 = readMatrix('q4_data/MATRIX.TRAIN.1400')
     # Train
    phi, mu_js, mu_jns = nb_train(dataMatrix_train1400, category_train1400)
    # Test and evluate
    prediction = nb_test(dataMatrix_test, phi, mu_js, mu_jns)
    print("Training size: 1400")
    e6 = 100*evaluate(prediction, category_test)

    plt.figure()
    plt.plot([50,100,200,400,800,1400], [e1, e2, e3, e4, e5, e6])
    plt.xlabel("Size of training data");plt.ylabel("Error %");
    plt.grid(True)
    plt.savefig("4c.png")
   
def main():
    # Note1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    # Note3: The shape of the data matrix (document matrix): (number of emails) by (number of tokens)

    # Load files
    dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')

    # Train
    phi, mu_js, mu_jns = nb_train(dataMatrix_train, category_train)

    # Test and evluate
    prediction = nb_test(dataMatrix_test, phi, mu_js, mu_jns)
    print("4a:")
    evaluate(prediction, category_test)

    # top5 spam words
    top5 = top5spam(mu_js, mu_jns, np.array(tokenlist))
    print("\n4b:Top 5 spam words\n",top5)

    # 4c
    print("\n4c:")
    multiData()
if __name__ == "__main__":
    main()
        
