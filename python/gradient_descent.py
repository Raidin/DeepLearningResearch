import numpy as np
import matplotlib.pyplot as plt

def QuadraticFunc(x, prime=False):
    if prime:
        return 2*x + 2
    else:
        return pow(x, 2) + 2 * x

def CubicFunc(x, prime=False):
    if prime:
        print(x)
        return 3 * pow(x, 2) + 6 * x - 6
    else:
        return  pow(x, 3) + 3 * pow(x, 2) - 6 * x - 8

# Learning Rate에 따라서 Local Minimum 찾는 속도 체크하기.
def gradient_descent(x, lr=0.1, step_num=100):
    x_history = []

    for i in range(step_num):
        x_history.append(x)

        grad = CubicFunc(x, True)
        x -= lr * grad

        if grad == 0: break

    return x, x_history

# tangent : 접선
def tangent_line(x):
    d = CubicFunc(x, True)
    y = CubicFunc(x) - d * x
    return lambda t: d * t + y

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111)

x = np.linspace(-4, 4, 100,True)
plt.plot(x, CubicFunc(x), label='function', linewidth='1.0', linestyle='-')

minimum, minimum_history = gradient_descent(-2.0)
print('Minimum Value :', minimum)
print('MiniMum History :', minimum_history)

# local minimum 그래프 그릴때 해당 지점 점으로 표시해주는 거 넣기..
for i, ele in enumerate(minimum_history[:10]) :
    tf = tangent_line(ele)
    plt.plot(x, tf(x), label='{}-th minimum({:.4f}) tangent line'.format(i, ele), linewidth='1.0', linestyle='--')

# Label Box Position
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.grid(alpha=.4,linestyle='--')
plt.ylim(-15, 15)

'''
axis 가운데로 정렬
'''
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.show()
