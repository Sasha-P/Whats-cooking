def cuisine_to_vector(cuisine_list_loc, cuisine):
    cuisine_vector = [0] * len(cuisine_list_loc)
    class_index = cuisine_list_loc.index(cuisine)
    cuisine_vector[class_index] = 1
    return cuisine_vector


def vector_to_cuisine(cuisine_list_loc, vector):
    class_index = vector.index(1)
    cuisine = cuisine_list_loc[class_index]
    return cuisine


def ingredients_to_vector(ingredients_list_loc, ingredients):
    ingredients_vector = [0] * len(ingredients_list_loc)
    for ingredient in ingredients:
        input_index = ingredients_list_loc.index(ingredient)
        ingredients_vector[input_index] = 1
    return ingredients_vector


def vector_to_ingredients(ingredients_list_loc, vector):
    ingredients = []
    indices = [i for i, x in enumerate(vector) if x == 1]
    for index in indices:
        ingredients.append(ingredients_list_loc[index])
    return ingredients
