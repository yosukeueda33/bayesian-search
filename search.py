# http://www2.stat.duke.edu/~banks/130-labs.dir/lab10.dir/Lab10_bayesian_search.pdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from random import randint
from random import random
from scipy.stats import multivariate_normal

size = (20,20)  # cols, rows
prob_of_detection = 0.1
noize_range = 5
# submarine_position = np.array([
#         randint(noize_range,size[0]-1-noize_range),
#         randint(noize_range,size[1]-1-noize_range)
# ])
submarine_position = np.array([16,10])
print(f"pos:{submarine_position}")

Y = np.zeros(size)
Y[submarine_position[0], submarine_position[1]] = 1

# expert_pred_center = submarine_position + np.array([
#     randint(-noize_range,noize_range),
#     randint(-noize_range,noize_range),
# ])
expert_pred_center = submarine_position + np.array([-3,5])

print(expert_pred_center)
x, y = np.mgrid[0:1:1.0/size[0], 0:1:1.0/size[1]]
pos = np.dstack((x, y))

pred_xy = expert_pred_center/np.array([size[1], size[0]])
# print(sub_pos_xy)
rv = multivariate_normal([pred_xy[1], pred_xy[0]], [[2.0, 0.3], [0.3, 0.5]])
pai = rv.pdf(pos)


def show_result(pai_in, search_index=None, savemode=False, fname=None):
    plt.clf()
    if savemode:
        matplotlib.use('Agg')
    ax = plt.subplot(1,1,1)
    axi = ax.imshow(pai_in)
    plt.colorbar(axi)
    ax.plot(submarine_position[0],submarine_position[1],"rx")
    # ax.plot(expert_pred_center[0],expert_pred_center[1],"go")
    if search_index is not None:
        ax.plot(search_index[0],search_index[1],"bo")
    
    if savemode:
        plt.savefig(fname)
    else:
        plt.show()

# https://www.haya-programming.com/entry/2018/02/06/152348
def p_select(index):

    if Y[index[0], index[1]] == 1:
        if prob_of_detection > random():
            return True

    return False

show_result(pai)

# https://qiita.com/tktktks10/items/f85aeef3321f6cbbd368
def get_max_index(nda):
    index = np.unravel_index(np.argmax(nda), nda.shape)
    return np.array([index[1], index[0]])

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

for i in range(0, 1000):
    search_index = get_max_index(pai)
    print(f"search:{search_index}")
    found = p_select(search_index)
    if found:
        print("found")
        break
    search_index_val_old = pai[search_index[1], search_index[0]]
    # Update pai elem(not searched).
    pai = pai / (1.0 - prob_of_detection*search_index_val_old)
    assert(pai.min() >= 0.0)

    # Update pai elem(searched).
    pai[search_index[1], search_index[0]] = (
        ((1.0 - prob_of_detection)*search_index_val_old) / \
        (1.0 - prob_of_detection*search_index_val_old)
    )

    # Normalize for avoiding saturation.
    pai = pai/pai.sum()
    print(f"max:{pai.max()}")


    padded_i = str(i).zfill(4)
    # show_result(pai, search_index)
    if i%5 ==0:
        show_result(
            pai, search_index,
            savemode=True, fname=f"out/{padded_i}.png")