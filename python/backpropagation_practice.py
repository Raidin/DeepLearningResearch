import numpy as np
import matplotlib.pyplot as plt

def MeanSquredError(y, t, prime=False):
    if prime:
        return y - t
    else:
        return np.sum(pow(t - y,2))/2

def Sigmoid(x, prime=False):
    if prime:
        return Sigmoid(x) * (1-Sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))

def Forward(W, x, act):
    return act(np.dot(x, W))


lr = 0.5
t = np.array([0.2, 0.7], dtype='float16')
x = np.array([0.7], dtype='float16')
w1 = np.array([[0.15], [0.2]], dtype='float16').reshape(1, 2)
w2 = np.array([[0.40, 0.50],
               [0.45, 0.55]], dtype='float16')

loss = []

for i in range(100):
    y1 = Forward(w1, x, Sigmoid)
    y2 = Forward(w2, y1, Sigmoid)
    loss.append(MeanSquredError(y2, t))

    # Error Function Derivative
    dz = MeanSquredError(y2, t, True)
    dy2 = Sigmoid(np.dot(y1, w2), True)

    dw2 = np.outer(y1, dy2) * dz
    print('dw2', dw2)
    dw1 = np.dot(w2, dy2 * dz) * Sigmoid(np.dot(x, w1), True) * x

    w2 -= lr * dw2
    w1 -= lr * dw1

    print('[{}]-iter loss :: {}'.format(i+1, loss[-1]))

print('- target :: ', t)
print('- output :: ', y2)
print('==================== W1 ====================\n', w1)
print('==================== W2 ====================\n', w2)

plt.plot(loss, color='red', linewidth='2.0', linestyle="--")
plt.grid(alpha=.9,linestyle='--')
plt.xlabel('iter')
plt.ylabel('loss')
plt.title('Loss')
# plt.show()
