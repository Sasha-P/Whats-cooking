from funcsEx04 import *
from utils import *


def train_nn(input_layer_size, hidden_layer_size, num_labels, X, y):
    # input_layer_size = 400
    # hidden_layer_size = 25
    # num_labels = 10

    Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
    Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    initial_nn_params = np.r_[
        np.reshape(initial_Theta1, Theta1.shape[0] * Theta1.shape[1], order='F'),
        np.reshape(initial_Theta2, Theta2.shape[0] * Theta2.shape[1], order='F')
    ]

    lamb = 1
    Theta = cgbt2(initial_nn_params, X, y, input_layer_size, hidden_layer_size, num_labels, lamb, 0.25, 0.5, 10, 1e-8)

    Theta1 = np.matrix(
        np.reshape(Theta[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1),
                   order='F'))
    Theta2 = np.matrix(
        np.reshape(Theta[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1), order='F'))

    p = predict(Theta1, Theta2, X)

    precision = 0
    for i in range(len(y)):
        if y[i] == p[i]:
            precision += 1

    print('Training Set Accuracy:', (1.0 * precision) / len(y))


if __name__ == '__main__':
    cuisine_list, ingredients_list, X, y = load_train()

    ingredients_count = len(ingredients_list)
    cuisines_count = len(cuisine_list)

    train_nn(ingredients_count, ingredients_count//8, cuisines_count, X, y)

    # print(len(cuisine_list))
    # print(len(ingredients_list))
    # print(len(X))
    # print(len(y))
