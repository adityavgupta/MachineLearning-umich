import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def gen_phi (x, M=2):
  N = x.shape[0]
  phi = np.zeros((N,M))

  for n in range(N):
    for j in range(M):
      phi[n,j] = x[n]**j
  return phi

def train_weights_unweighted(x, y, M):
  N = x.shape[0]
  phi = gen_phi(x, M)
  yt = y.reshape((y.shape[0],1))
  w = np.matmul(np.linalg.pinv(phi), yt)
  return w

def train_weights_weighted(x,y,R, M):
  phi = gen_phi(x, M)
  yt = y.reshape((y.shape[0],1))
  w = np.linalg.inv(phi.T @ (R @ phi)) @(phi.T @ (R @ yt))
  return w

def gen_R(x, xq, t):
  r = np.zeros(x.shape[0])
  for i in range(len(r)):
    r[i] = np.exp(-1*((xq-x[i])**2)/(2*(t**2)))
  R = np.diag(r, k=0)
  return R

def locally_weighted_linear_regression(x,y,t):
  x_query = np.linspace(min(x), max(x), num=100)
  pred = np.zeros(x_query.shape[0])
  for i in range(len(x_query)):
    R = gen_R(x,x_query[i],t)
    w = train_weights_weighted(x,y,R,M=2)
    pred[i] = gen_phi(x_query[i], M=2) @ w
  return x_query, pred


def main():
  q2x = np.load('q2x.npy')
  q2y = np.load('q2y.npy')

  X = q2x.reshape(q2x.shape[0],1)
  y_vec = q2y.reshape(q2y.shape[0],1)

  # Question 2d i
  u_w = train_weights_unweighted(X, y_vec, M=2)
  pred_1 = np.matmul(gen_phi(X, M=2), u_w)
  p = q2x.argsort()
  pred_1t = pred_1[:,0][p]
  plt.figure(1)
  plt.plot(q2x[p], pred_1t, color="#9ACD32", label="unweighted linear regression")
  plt.text(2.5,1,"y = %.3fx + %.3f"%(u_w[0,0], u_w[1,0]))
  plt.scatter(q2x, q2y)
  plt.xlabel("x"); plt.ylabel("y")
  plt.title("Unweighted linear regression")
  plt.grid(True)
  plt.savefig("2d-i.png")

  # Question 2d ii
  x_query, lwlr_pred = locally_weighted_linear_regression(X, y_vec, 0.8)
  
  xq_sorted_indices = x_query[:,0].argsort()
  x_points = x_query[:,0][xq_sorted_indices]
  lwlr_pred_sorted = lwlr_pred[xq_sorted_indices]
  plt.figure(2)
  plt.plot(x_points, lwlr_pred_sorted, color="#F97306", label="locally weighted linear regression")
  plt.scatter(q2x, q2y)
  plt.grid(True)
  plt.xlabel("x"); plt.ylabel("y")
  plt.title("Locally weighted linear regression")
  plt.legend(loc='best')
  plt.savefig("2d-ii.png")

  # Question 2d iii
  plt.figure(3)
  T = [0.1, 0.2, 3, 10]
  color = iter(cm.rainbow(np.linspace(0, 1, 4)))
  for t in T:
    x_query, lwlr_pred = locally_weighted_linear_regression(X, y_vec, t)
  
    xq_sorted_indices = x_query[:,0].argsort()
    x_points = x_query[:,0][xq_sorted_indices]
    lwlr_pred_sorted = lwlr_pred[xq_sorted_indices]
    c = next(color)
    plt.plot(x_points, lwlr_pred_sorted, c=c, label=r'$\tau$ =%0.1f'%(t))

  plt.scatter(q2x, q2y, color="orange", alpha=0.3, label="Data")
  plt.grid(True)
  plt.xlabel("x"); plt.ylabel("y")
  plt.title(r'Locally Weighted Linear Regression for different values of $\tau$')
  plt.legend(loc='best')
  plt.savefig("2d-iii.png")
  plt.show()

if __name__== "__main__":
  main()