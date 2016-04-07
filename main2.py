from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from utils import *


cuisine_list, ingredients_list, x, y = load_train()
classifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x, y)
p = classifier.predict(x)

precision = 0
for i in range(len(y)):
    if y[i] == p[i]:
        precision += 1
accuracy = (1.0 * precision) / len(y)

print('Training Set Accuracy:', accuracy)

t, ids = load_test(ingredients_list)
p = classifier.predict(t)
save_result(cuisine_list, p, ids)


