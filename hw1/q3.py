import numpy as np
import matplotlib.pyplot as plt

LR = .1 # learning rate

def activate_fn(n):
    return 1 if n>=0 else 0

def plot_fn(x, w_all, output_all=True):
    for j in range(len(w_all)):
        if not output_all:
            j = len(w_all) - 1
        xx = np.linspace(-0.5, 1.5, 20)
        yy = -(w_all[j][0] + w_all[j][1]*xx) / w_all[j][2]
        plt.figure()
        plt.plot(xx, yy)
        plt.scatter(0, 0, color='r')
        plt.scatter(0, 1, color='r')
        plt.scatter(1, 0, color='r')
        plt.scatter(1, 1, color='b')
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.axis('equal')
        plt.title('Iteration:{}/{}'.format(j+1, len(w_all)))
        plt.grid()
        plt.show()
        if not output_all:
            return 0

def plot_tr(w_all):
    n = len(w_all)
    plt.figure()
    for i in range(3):
        plt.plot(list(range(n)), [w_all[j][i] for j in range(n)])
    plt.legend(['w0','w1','w2'])
    plt.title('Parameters trajectory')
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.show()

if __name__ == '__main__':
    w = np.random.uniform(-1, 1, size=3)
    w_rollout = []
    x = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    y = np.array([0, 0, 0, 1]) # and
    # y = np.array([0, 1, 1, 1]) # or
    # y = np.array([1, 1, 1, 0]) # nand
    # y = np.array([0, 1, 1, 0]) # xor
    while True:
        w_rollout.append(w.copy())
        y_predict = np.squeeze(np.asarray([activate_fn(w.dot(x[i])) for i in range(4)]))
        err = y - y_predict
        delta_w = LR * np.array([err[i] * x[i] for i in range(4)])
        w += np.sum(delta_w, axis=0)
        if np.sum(np.square(err)) == 0 or len(w_rollout) > 100:
            break
    print('Weights:{}\nNumber of iters:{}'.format(w, len(w_rollout)))
    plot_fn(x, w_rollout, output_all=False)
    plot_tr(w_rollout)
    
