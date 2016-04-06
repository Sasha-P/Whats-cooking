import json
from utils import cuisine_to_vector, vector_to_cuisine, ingredients_to_vector, vector_to_ingredients


def load_data(file_path_loc):
    training_file = open(file_path_loc)
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

    for dish in training_data:
        cuisine = dish['cuisine']
        ingredients = dish['ingredients']
        dish['cuisine_vector'] = cuisine_to_vector(cuisine_list_loc, cuisine)
        dish['ingredients_vector'] = ingredients_to_vector(ingredients_list_loc, ingredients)

    return cuisine_list_loc, ingredients_list_loc


if __name__ == '__main__':
    file_path = 'data/train.json'
    cuisine_list, ingredients_list = load_data(file_path)

    # print(len(cuisine_list))
    # print(len(ingredients_list))
