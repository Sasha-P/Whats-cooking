import json
import csv
import time
import numpy as np


def load_train():
    file_path = 'data/train.json'
    training_file = open(file_path)
    training_data = json.load(training_file)

    cuisine_list_loc = []
    ingredients_list_loc = []

    for dish in training_data:
        cuisine = dish['cuisine']
        ingredients = dish['ingredients']
        if cuisine not in cuisine_list_loc:
            cuisine_list_loc.append(cuisine)

        ingredients_list_loc += ingredients
        ingredients_set_loc = set(ingredients_list_loc)
        ingredients_list_loc = list(ingredients_set_loc)

    cuisine_list_loc.sort()
    ingredients_list_loc.sort()

    x = []
    y = []

    for dish in training_data:
        cuisine = dish['cuisine']
        ingredients = dish['ingredients']
        cuisine_number = cuisine_to_number(cuisine_list_loc, cuisine)
        ingredients_vector = ingredients_to_vector(ingredients_list_loc, ingredients)
        y.append(cuisine_number)
        x.append(ingredients_vector)

    print('Loaded', file_path, 'with', len(training_data), 'examples')

    return cuisine_list_loc, ingredients_list_loc, np.array(x), np.array(y)


def load_test(ingredients_list):
    file_path = 'data/test.json'
    test_file = open(file_path)
    test_data = json.load(test_file)

    x = []
    ids = []

    for dish in test_data:
        ingredients = dish['ingredients']
        ingredients_vector = ingredients_to_vector(ingredients_list, ingredients)
        x.append(ingredients_vector)
        ids.append(dish['id'])

    print('Loaded', file_path, 'with', len(test_data), 'examples')

    return np.array(x), ids


def save_result(cuisine_list, prediction, ids):
    with open('results/' + time.strftime("%Y.%m.%d_%H.%M.%S") + '_submission.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['id', 'cuisine'])
        for i in range(len(ids)):
            spamwriter.writerow([ids[i], number_to_cuisine(cuisine_list, prediction[i])])
    pass


def cuisine_to_number(cuisine_list_loc, cuisine):
    """
    Convert cuisine to number representation

    :param cuisine_list_loc: list of str of all cuisines
    :param cuisine: str that represent cuisine
    :return: int number of cuisine

    >>> cuisine_to_number(['1', '2', '3'], '2')
    1
    >>> cuisine_to_number(['1', '3', '5'], '1')
    0
    >>> cuisine_to_number(['1', '3', '5', '7'], '5')
    2
    """
    class_index = cuisine_list_loc.index(cuisine)
    return class_index


def number_to_cuisine(cuisine_list_loc, number):
    """
    Convert number to cuisine representation

    :param cuisine_list_loc: list of str of all cuisines
    :param number: int number of cuisine
    :return: str that represent cuisine

    >>> number_to_cuisine(['1', '2', '3'], 1)
    '2'
    >>> number_to_cuisine(['1', '3', '5'], 0)
    '1'
    >>> number_to_cuisine(['1', '3', '5', '7'], 2)
    '5'
    """
    cuisine = cuisine_list_loc[number]
    return cuisine


def cuisine_to_vector(cuisine_list_loc, cuisine):
    """
    Convert cuisine to vector representation

    :param cuisine_list_loc: list of str of all cuisines
    :param cuisine: str that represent cuisine
    :return: list of int where cuisine is 1 other 0

    >>> cuisine_to_vector(['1', '2', '3'], '2')
    [0, 1, 0]
    >>> cuisine_to_vector(['1', '3', '5'], '1')
    [1, 0, 0]
    >>> cuisine_to_vector(['1', '3', '5', '7'], '5')
    [0, 0, 1, 0]
    """
    cuisine_vector = [0] * len(cuisine_list_loc)
    class_index = cuisine_list_loc.index(cuisine)
    cuisine_vector[class_index] = 1
    return cuisine_vector


def vector_to_cuisine(cuisine_list_loc, vector):
    """
    Convert cuisine vector to cuisine

    :param cuisine_list_loc: list of str of all cuisines
    :param vector: list of int where cuisine is 1 other 0
    :return: str that represent cuisine

    >>> vector_to_cuisine(['1', '2', '3'], [0, 1, 0])
    '2'
    >>> vector_to_cuisine(['1', '2', '3'], [1, 0, 0])
    '1'
    >>> vector_to_cuisine(['1', '3', '5', '7'], [0, 0, 1, 0])
    '5'
    """
    class_index = vector.index(1)
    cuisine = cuisine_list_loc[class_index]
    return cuisine


def ingredients_to_vector(ingredients_list_loc, ingredients):
    """
    Convert ingredients to vector representation

    :param ingredients_list_loc: list of str of all ingredients
    :param ingredients: list of str of ingredients
    :return: list of int where ingredients is 1 other 0

    >>> ingredients_to_vector(['1', '2', '3', '4', '5', '6', '7'], ['1', '3', '7'])
    [1, 0, 1, 0, 0, 0, 1]
    >>> ingredients_to_vector(['1', '2', '3', '4', '5', '6', '7'], ['1', '7'])
    [1, 0, 0, 0, 0, 0, 1]
    >>> ingredients_to_vector(['1', '2', '3', '4', '5', '6', '7'], ['5'])
    [0, 0, 0, 0, 1, 0, 0]
    >>> ingredients_to_vector(['1', '2', '3', '4', '5', '6', '7'], ['8'])
    [0, 0, 0, 0, 0, 0, 0]
    """
    ingredients_vector = [0] * len(ingredients_list_loc)
    for ingredient in ingredients:
        if ingredient in ingredients_list_loc:
            input_index = ingredients_list_loc.index(ingredient)
            ingredients_vector[input_index] = 1
    return ingredients_vector


def vector_to_ingredients(ingredients_list_loc, vector):
    """
    Convert ingredients vector to ingredients

    :param ingredients_list_loc: list of str of all ingredients
    :param vector: list of int where ingredients is 1 other 0
    :return: list of str of ingredients

    >>> vector_to_ingredients(['1', '2', '3', '4', '5', '6', '7'], [1, 0, 1, 0, 0, 0, 1])
    ['1', '3', '7']
    >>> vector_to_ingredients(['1', '2', '3', '4', '5', '6', '7'], [1, 0, 0, 0, 0, 0, 1])
    ['1', '7']
    >>> vector_to_ingredients(['1', '2', '3', '4', '5', '6', '7'], [0, 0, 0, 0, 1, 0, 0])
    ['5']
    """
    ingredients = []
    indices = [i for i, x in enumerate(vector) if x == 1]
    for index in indices:
        ingredients.append(ingredients_list_loc[index])
    return ingredients
