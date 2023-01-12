from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
from random import randint
import warnings

warnings.filterwarnings('ignore')

mn = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mn.load_data()
X_train = np.array(X_train, dtype='float64')
X_test = np.array(X_test, dtype='float64')

X_train = X_train[:60000]
y_train = y_train[:60000]
X_test = X_test[:10000]

data_train = {'images': X_train, 'labels': y_train}

data_test = {'images': X_test, 'labels': y_test}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fsum(a, b):
    return np.sum(a * b)


def neuron_sig(x, w):
    return sigmoid(fsum(x, w))


# Возвращает вычисленную цифру
def get_number(x_arr, w_arr):
    max_idx = -1
    max_value = -1
    for i in range(10):
        val = neuron_sig(x_arr, w_arr[i])
        if val > max_value:
            max_idx = i
            max_value = val
    if max_idx == -1:
        return randint(0, 9)
    else:
        return max_idx


w = np.empty((10, 28, 28), dtype='float64')
# Считываем веса из файлов
for tr_n in range(10):
    with open('./model' + str(tr_n) + '.w', 'r') as file:
        for i in range(0, 20):
            for j in range(0, 20):
                w[tr_n][i][j] = file.readline()

test = np.empty((len(X_test), 10), dtype='float32')

for i in range(len(X_test)):
    for j in range(10):
        x_vec = np.array(X_test[i])
        test[i][j] = neuron_sig(x_vec, w[j])

test_count = len(X_test)

un_s_test = []
s_test = 0  # Счетчик правильных ответов
for i in range(test_count):
    x_vec = np.array(X_test[i])
    calc_num = get_number(x_vec, w)
    if calc_num == y_test[i]:
        s_test += 1
    else:
        un_s_test.append(i)

print('Успешно распознано:', s_test, '-', round(100 * s_test / test_count, 3), '%')

print('Нераспознаны изображения:', un_s_test)

for idx in range(len(un_s_test)):
    print('idx:', un_s_test[idx])
    print('Цифра:', y_test[un_s_test[idx]])
    print('Ответ сети:', test[un_s_test[idx]])
    plt.imshow(data_test['images'][un_s_test[idx]], cmap='gray')
    plt.show()
