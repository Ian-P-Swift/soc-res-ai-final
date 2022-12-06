import data
import models
import evaluation

PERCENT = 1
LEGACY = True
SMOTE = True
USE_INFLUENCE = True
FILTER_FACES = True

d = data.Data()
d.load_images('./train/')
if FILTER_FACES:
    d.load_faces()
if USE_INFLUENCE:
    if LEGACY:
        d.load_filters('uploss_scores_skin_tone_legacy.dill', 'uploss_scores_gender_legacy.dill', 'uploss_scores_age_legacy.dill', PERCENT)
    else:
        d.load_filters('uploss_scores_skin_tone.dill', 'uploss_scores_gender.dill', 'uploss_scores_age.dill', PERCENT)

if FILTER_FACES:
    X, Y = d.get_faces(influence_filter=USE_INFLUENCE)
else:
    X, Y = [d.X for i in range(3)], d.Y
    
if SMOTE:
    X, Y = d.smote_data(X, Y)

facial_categorizer = models.FacialCategoriesModel(X, Y)
models.train_facial_categorical_model(facial_categorizer)

faces_x, faces_y = d.split_faces()
face_detection = models.FacialDetectionModel(faces_x, faces_y)
face_detection.fit(epochs=35, batch_size=32, save=True, save_location='./models/', verbose=True)

combined_model = models.CombinedModel(facial_categorizer, face_detection)

d_test = data.Data()
d_test.load_images('./test/')

X,Y = d_test.X, d_test.Y

pred = facial_categorizer.predict(X)

accuracies = evaluation.model_accuracy(pred, Y)
class_accuraciess = evaluation.class_accuracies(pred, Y)
disparities = evaluation.disparities(class_accuraciess)

print('Classification Accuracy, Class Accuracies, and Disparity for Skin Tone')
print(accuracies[0])
print(class_accuraciess[0])
print(disparities[0])
print('--------------------\n')

print('Classification Accuracy, Class Accuracies, and Disparity for Gender')
print(accuracies[1])
print(class_accuraciess[1])
print(disparities[1])
print('--------------------\n')


print('Classification Accuracy, Class Accuracies, and Disparity for Age')
print(accuracies[2])
print(class_accuraciess[2])
print(disparities[2])
print('--------------------\n')

