import json

training_file = open('data/train.json')
training_data = json.load(training_file)
# print(training_data[0])

cuisine_list = []
ingredients_list = []
for dish in training_data:
    cuisine = dish['cuisine']
    ingredients = dish['ingredients']
    if cuisine not in cuisine_list:
        cuisine_list.append(cuisine)

    ingredients_list += ingredients
    ingredients_set = set(ingredients_list)
    ingredients_list = list(ingredients_set)

print(len(cuisine_list))
print(len(ingredients_list))

# ingredients
# cuisine
