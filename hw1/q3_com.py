import numpy as np
import matplotlib.pyplot as plt

LR = 1

def activate_fn(n):
    return 1 if n>=0 else 0

def plot_tr(w_all):
    n = len(w_all)
    plt.figure()
    for i in range(2):
        plt.plot(list(range(n)), [w_all[j][i] for j in range(n)])
    plt.legend(['w0','w1'])
    plt.title('Parameters trajectory')
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.show()

if __name__ == '__main__':
    w = np.random.uniform(-1, 1, size=2)
    w_rollout = []
    x = np.array([[1, 0], [1, 1]])
    y = np.array([1, 0])
    while True:
        w_rollout.append(w.copy())
        y_predict = np.squeeze(np.asarray([activate_fn(w.dot(x[i])) for i in range(2)]))
        err = y - y_predict
        delta_w = LR * np.array([err[i] * x[i] for i in range(2)])
        w += np.sum(delta_w, axis=0)
        if np.sum(np.square(err)) == 0:
            break
    print('Weights:{}\nNumber of iters:{}'.format(w, len(w_rollout)))
    plot_tr(w_rollout)
    
