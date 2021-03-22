import numpy as np
import matplotlib.pyplot as plt

LR = .1
NUM_EPOCH = 100

def plot_fn(w):
    x = np.linspace(-1, 6, 100)
    y0 = 8.868 - 1.636 * x
    xs = [0.5, 1.5, 3, 4, 5]
    ys = [8, 6, 5, 2, 0.5]
    y = w[0] + w[1]*x
    plt.figure()
    plt.scatter(xs, ys)
    plt.plot(x, y0)
    plt.plot(x, y)
    plt.legend(['LLS', 'LMS'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear regression')
    plt.show()

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

def plot_loss(err_all):
    plt.figure()
    plt.plot(err_all)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Regression loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    w = np.random.uniform(-10, 10, size=2)
    w_rollout = []
    err_rollout = []
    x = np.array([[1, 0.5], [1, 1.5], [1, 3], [1, 4], [1, 5]])
    y = np.array([8, 6, 5, 2, 0.5])
    for _ in range(NUM_EPOCH):
        w_rollout.append(w.copy())
        y_predict = np.squeeze(np.asarray([w.dot(x[i]) for i in range(y.shape[0])]))
        err = y - y_predict
        err_rollout.append(np.sum(err))
        delta_w = LR * np.array([err[i] * x[i] for i in range(y.shape[0])])
        w += np.sum(delta_w, axis=0)

    print('Weights:{}'.format(w))
    plot_fn(w_rollout[-1])
    plot_tr(w_rollout)
    plot_loss(err_rollout)