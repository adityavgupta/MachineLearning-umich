import numpy as np
import matplotlib.pyplot as plt


# Generate the matrix of basis functions
# takes in the data and the degree of the polynomial
def gen_phi (x, M=2):
  N = x.shape[0]
  phi = np.zeros((N,M))

  for n in range(N):
    for j in range(M):
      phi[n,j] = x[n]**j
  return phi

def BGD(x, y, M, lr, nIters):
    phi = gen_phi(x, M)
    yt=y.reshape((20,1))
    N = x.shape[0]
    w = np.zeros((M,1))
    err = np.zeros(nIters)
    flag = 1
    for num in range(nIters):
        temp_grad = 0
        for n in range(N):
            temp_grad += (np.matmul(w.T, phi[n,:].reshape(-1,1)) - y[n])*phi[n,:].reshape(-1,1)
        
        pred = np.matmul(gen_phi(x, M=M), w)
        sse = np.sum((pred-yt)**2)/len(pred)
        w = w - lr*(temp_grad)
        if sse < 0.2 and flag==1:
            print("Num epochs for BGD convergence:", num+1)
            flag=0
        err[num] = sse
    
    return w,err

def SGD(x, y, M, lr, k, nIters):
    phi = gen_phi(x, M)
    yt=y.reshape((20,1))
    N = x.shape[0]
    w = np.zeros((M,1))
    err = np.zeros(nIters)
    a= lr
    flag = 1
    for num in range(nIters):
        temp_grad = 0
        for n in range(N):
            temp_grad = (np.matmul(w.T, phi[n,:].reshape(-1,1)) - y[n])*phi[n,:].reshape(-1,1)
            w = w - a*(temp_grad)
        a = k*lr/np.sqrt(num+0.1)
        pred = np.matmul(gen_phi(x, M=M), w)
        sse = np.sum((pred-yt)**2)/len(pred)
        if sse < 0.2 and flag==1:
            print("Num epochs for SGD convergence:", num+1)
            flag = 0
        
        err[num] = sse
        
    return w,err

def Erms(x, y, w, M):
    N = x.shape[0]
    phi = gen_phi(x, M)
    yt = y.reshape((20, 1))
    E_w = np.matmul(w.T, np.matmul(phi.T,(0.5*np.matmul(phi, w) - yt))) + 0.5*np.matmul(yt.T, yt)
    Erms = np.sqrt(2*E_w/N)
    return Erms

def train_weights_closed_form(x, y, M):
    N = x.shape[0]
    phi = gen_phi(x, M)
    yt = y.reshape((y.shape[0],1))
    w = np.matmul(np.linalg.pinv(phi), yt)
    return w


def gen_erms_M(x_train, y_train, x_test, y_test, M_vals):
    errors_train = np.zeros(M_vals.shape[0])
    errors_test = np.zeros(M_vals.shape[0])
    
    for m in M_vals:
        w = train_weights_closed_form(x_train, y_train, m)
        errors_train[m-1] = Erms(x_train, y_train, w, m)
        errors_test[m-1] = Erms(x_test, y_test, w, m)
        
    return errors_train, errors_test

def train_weight_regularized(x, y, M, lam):
    N = x.shape[0]
    phi = gen_phi(x,M)
    yt = y.reshape((y.shape[0], 1))
    phiTphi = np.matmul(phi.T, phi)
    inv_term = np.linalg.inv(lam*np.identity(phiTphi.shape[0]) + phiTphi)
    w = np.matmul(inv_term, np.matmul(phi.T, yt))
    return w

def gen_erms_lambda(x_train, y_train, x_test, y_test, lambda_vals):
    errors_train = np.zeros(lambda_vals.shape[0])
    errors_test = np.zeros(lambda_vals.shape[0])
    weights = []
    for i in range(len(lambda_vals)):
        w = train_weight_regularized(x_train, y_train, M=10, lam=lambda_vals[i])
        weights.append(w)
        errors_train[i] = Erms(x_train, y_train, w, M=10)
        errors_test[i] = Erms(x_test, y_test, w, M=10)
    
    return weights, errors_train, errors_test

def main():
    
    x_test = np.load('q1xTest.npy')
    x_train = np.load('q1xTrain.npy')
    y_train = np.load('q1yTrain.npy')
    y_test = np.load('q1yTest.npy')

    ## Question 1 a
    print("** Question 1 a **")
    iters = 300
    lr = 0.017
    k = 7
    w_batch,err_batch = BGD(x_train, y_train, M=2, lr=lr, nIters=iters)
    w_sgd, err_sgd = SGD(x_train, y_train, M=2, lr=lr, k=k, nIters=iters)

    print("Hyperparams:\n initial learning rate=%.3f\n iterations=%d\n k=%d" %(lr, iters, k))
    print("Batch GD coefficients:\n", w_batch)
    print("Batch GD error rate:\n", err_batch[-1])
    print("Stochastic GD coefficients:\n", w_sgd)
    print("Stochastic GD error rate:\n%f\n" % (err_sgd[-1]))
    # plots for batch gradient
    plt.figure(1, (9,5))
    plt.subplot(1,2,1)
    plt.scatter(x_test, y_test)
    y_pred_b =  np.matmul(gen_phi(x_test, M=2),w_batch)
    y_pred_sgd =  np.matmul(gen_phi(x_test, M=2),w_sgd)
    plt.plot(x_test,y_pred_b, label="BGD")
    plt.plot(x_test,y_pred_sgd, label="SGD")
    plt.title("Best fit using BGD and SGD")
    plt.xlabel("Data"); plt.ylabel(r'y = $\Phi w$')
    plt.legend(loc='best')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(np.arange(0,iters), err_batch, label="BGD")
    plt.plot(np.arange(0,iters), err_sgd, label="SGD")
    plt.title("Mean Square error for BGD and SGD")
    plt.xlabel("Epochs"); plt.ylabel(r'$E_{MS}$')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig("q1-a.png")

    ## Question 1b
    M_vals = np.arange(1,11)
    Erms_vals_train, Erms_vals_test = gen_erms_M(x_train, y_train, x_test, y_test, M_vals)
    plt.figure(2, (8,6))
    print("** Question 1b (see plot) **\n")
    plt.plot(M_vals-1, Erms_vals_train, '-b', marker='o', label='Training data')
    plt.plot(M_vals-1, Erms_vals_test, '-r', marker='o', label='Testing data')
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Erms")
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig("q1-b.png")
    
    ## Question 1c
    lam_vals = np.array([10**-14, 10**-8, 10**-7, 10**-6,10**-5,10**-4,10**-3,10**-2,10**-1, 1])
    weights, Erms_train_1c, Erms_test_1c = gen_erms_lambda(x_train, y_train, x_test, y_test, lam_vals)
    min_test_err_idx = np.argmin(Erms_test_1c)
    plt.figure(3, (8,6))
    print("** Question 1c **")
    print("Lambda value with least test error:", lam_vals[min_test_err_idx]," ln(lambda):",np.log(lam_vals[min_test_err_idx]), "\n")
    
    for i in range(len(lam_vals)):
        print("lambda: ",lam_vals[i], "ln(lambda): ", np.round(np.log(lam_vals[i]),4), "\nw:\n",np.round(weights[i],4),"\n")
    plt.plot(np.log(lam_vals), Erms_train_1c, '-b', marker='o', label='Training data')
    plt.plot(np.log(lam_vals), Erms_test_1c, '-r', marker='o', label='Testing data')
    plt.xlabel(r'ln $\lambda$')
    plt.ylabel("Erms")
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig("q1-c.png")
    plt.show()
if __name__ == "__main__":
    main()





