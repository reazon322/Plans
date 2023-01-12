from tensorflow import keras
import numpy as np
from random import randint
import warnings

warnings.filterwarnings("ignore")

mn = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mn.load_data()

X_train = np.array(X_train, dtype='float64')
X_test = np.array(X_test, dtype='float64')

X_train = X_train[:60000]
y_train = y_train[:60000]
X_test = X_test[:10000]

data_train = {'images': X_train, 'labels': y_train}

data_test = {'images': X_test, 'labels': y_test}


# Функция суммирования
def fsum(a, b):
    return np.sum(a * b)


# Функция активации (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def neuron_sig(x, w):
    return sigmoid(fsum(x, w))


def bin_files(tr_num, w):
    with open("./model" + str(tr_num) + ".w", 'w') as file:
        for i in range(0, 28):
            for j in range(0, 28):
                file.write(str(w[i][j]) + '\n')



train_len = len(y_train)

x = X_train / 255

w = np.empty((10, 28, 28), dtype='float64')
dw = np.empty((28, 28), dtype='float64')
y = np.empty(train_len, dtype='float64')
err = np.empty(train_len, dtype='float64')

# Скорость обучения
bias = 0.5

max_iteration = 1000  # Попытка охватить все пикчи

for train_number in range(10):
    for i in range(train_len):
        if y_train[i] == train_number:
            y[i] = 1
        else:
            y[i] = 0  # 1 - норм, 0 - нет
            
    for i in range(0, 28):
        for j in range(0, 28):
            w[train_number][i][j] = randint(-1, 1)
            dw[i][j] = 0
    counter = 0
    min_mse = 1
    min_mse_w = w[train_number]

    while True:
        dw = dw * 0
        for i in range(train_len):
            x_vec = np.array(x[i])
            y_pred = neuron_sig(x_vec, w[train_number])
            d_f = y_pred * (1 - y_pred)
            err[i] = y[i] - y_pred
            dw = dw * 0
            dw = dw + -2 * err[i] * x[i] * d_f
            #print(dw)
            w[train_number] = w[train_number] - dw * bias

        for i in range(train_len):
            x_vec = np.array(x[i])
            y_pred = neuron_sig(x_vec, w[train_number])
            d_f = y_pred * (1 - y_pred)
            err[i] = y[i] - y_pred
        mse = np.sum(err ** 2) / train_len

        if mse < min_mse:
            min_mse = mse
            min_mse_w = w[train_number]

        if mse < 0.001:
            break
        counter += 1
        if counter > max_iteration:
            break
    bin_files(train_number, min_mse_w)
    print(f"Цифра: {train_number} 'Кол-во итераций =' {counter} 'MSE =' {mse}")
