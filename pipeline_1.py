import numpy as np

import data
import models
import evaluation

d = data.Data()
d.load_images('./train/')
# d.learn_faces(.9)
d.load_faces()
X, Y = d.get_faces()

facial_categorizer = models.FacialCategoriesModel(X, Y)
models.train_facial_categorical_model(facial_categorizer)
pred = facial_categorizer.predict(X[0])

accs = evaluation.class_accuracies(pred, Y)
min_acc_class = [evaluation.min_accuracy(accs[i]) for i in range(len(accs))]
mislabeled = evaluation.failure_positions(pred, Y, min_acc_class)


print('MISLABELED SKIN TONE PICTURES IN LOWEST ACCURACY CLASS:')
print([d.get_filename_for_image(X[0][i]) for i in mislabeled[0]])
print('--------------------\n')

print('MISLABELED GENDER PICTURES IN LOWEST ACCURACY CLASS:')
print([d.get_filename_for_image(X[1][i]) for i in mislabeled[1]])
print('--------------------\n')

print('MISLABELED AGE PICTURES IN LOWEST ACCURACY CLASS:')
print([d.get_filename_for_image(X[2][i]) for i in mislabeled[2]])
