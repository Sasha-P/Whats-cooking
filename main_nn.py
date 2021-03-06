import funcsEx04 as fnx
import utils as utl
import numpy as np


def train_nn(input_layer_size, hidden_layer_size, num_labels, X, y):
    Theta1 = fnx.randInitializeWeights(input_layer_size,hidden_layer_size)
    Theta2 = fnx.randInitializeWeights(hidden_layer_size, num_labels)

    initial_Theta1 = fnx.randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = fnx.randInitializeWeights(hidden_layer_size, num_labels)

    initial_nn_params = np.r_[
        np.reshape(initial_Theta1, Theta1.shape[0] * Theta1.shape[1], order='F'),
        np.reshape(initial_Theta2, Theta2.shape[0] * Theta2.shape[1], order='F')
    ]

    lamb = 1
    Theta = fnx.cgbt2(initial_nn_params, X, y, input_layer_size, hidden_layer_size, num_labels, lamb, 0.25, 0.5, 10, 1e-8)

    Theta1 = np.matrix(
        np.reshape(Theta[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1), order='F'))
    Theta2 = np.matrix(
        np.reshape(Theta[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1), order='F'))

    p = fnx.predict(Theta1, Theta2, X)

    precision = 0
    for i in range(len(y)):
        if y[i] == p[i]:
            precision += 1

    print('Training Set Accuracy:', (1.0 * precision) / len(y))

    return Theta1, Theta2


if __name__ == '__main__':
    cuisine_list, ingredients_list, X, y = utl.load_train('number')

    ingredients_count = len(ingredients_list)
    cuisines_count = len(cuisine_list)

    Theta1, Theta2 = train_nn(ingredients_count, ingredients_count//16, cuisines_count, X, y)

    T, ids = utl.load_test(ingredients_list)

    p = fnx.predict(Theta1, Theta2, T)

    utl.save_result('nn', cuisine_list, p, ids)

