###SOME CODE PROVIDED AS BOILERPLATE CODE FOR EIGHT BIT BIAS BOUNTY###

import os
from threading import Thread

import numpy as np
from keras.applications import EfficientNetV2B0
from keras.callbacks import EarlyStopping
import keras as K


class FacialCategoriesModel():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.trainX, self.testX, self.trainY, self.testY = train_validation_split(X, Y)
        self.cat = ['skin_tone', 'gender', 'age']
        self.loss = ['categorical_crossentropy' for i in range(3)]
        self.metrics = [['accuracy'] for i in range(3)]
        self.models = [None] * 3
        self.save_path = './models/'

    # train a model specific for a certain class index in self.cat
    def fit(self, index, epochs=5, batch_size=32, save=False, save_location=None, verbose=1):
        model = initializeModel()
        model.add(K.layers.Dense(self.trainY[index].shape[1], activation='softmax'))
        es = EarlyStopping(monitor="val_accuracy", mode="max", patience=10)
        model.compile(loss=self.loss[index], optimizer='Adam', metrics=self.metrics[index])
        model.fit(
            self.trainX[index], self.trainY[index],
            validation_data=(self.testX[index], self.testY[index]),
            batch_size=batch_size, epochs=epochs, verbose=verbose,
            callbacks=[es]
        )
        if save:
            if os.path.exists(self.save_path) == False:
                print('save location ' + self.save_path + ' did not exist. creating')
                os.makedirs(self.save_path)
            SAVE_LOCATION = save_location + 'model_' + self.cat[index] + '.h5'
            print("saving model at " + SAVE_LOCATION)
            model.save(SAVE_LOCATION)
        self.models[index] = model

    def predict(self, newX):
        predictions = [self.models[model].predict(newX) for model in range(3)]

        return predictions

class FacialDetectionModel():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.trainX, self.testX, self.trainY, self.testY = train_validation_split_single(X, Y)
        self.loss = 'categorical_crossentropy'
        self.metric = ['accuracy']
        self.model = None
        self.save_path = './models/'

    # train a model specific for a certain class index in self.cat
    def fit(self, epochs=5, batch_size=32, save=False, save_location=None, verbose=1):
        model = initializeModel()
        model.add(K.layers.Dense(self.trainY.shape[1], activation='softmax'))
        es = EarlyStopping(monitor="val_accuracy", mode="max", patience=10)
        model.compile(loss=self.loss, optimizer='Adam', metrics=self.metric)
        model.fit(
            self.trainX, self.trainY,
            validation_data=(self.testX, self.testY),
            batch_size=batch_size, epochs=epochs, verbose=verbose,
            callbacks=[es]
        )
        if save:
            if os.path.exists(self.save_path) == False:
                print('save location ' + self.save_path + ' did not exist. creating')
                os.makedirs(self.save_path)
            SAVE_LOCATION = save_location + 'model_fnf.h5'
            print("saving model at " + SAVE_LOCATION)
            model.save(SAVE_LOCATION)
        self.model = model

    def predict(self, newX):
        predictions = self.model.predict(newX)

        return predictions

class CombinedModel:
    def __init__(self, face_categorizer, face_detector):
        self.face_categorizer = face_categorizer
        self.face_detector = face_detector

    def predict(self, newX):
        fnf_preds = self.face_detector.predict(newX)
        fnf_preds = np.argmax(fnf_preds, axis=1)

        predictions = self.face_categorizer.predict(newX[np.where(fnf_preds == 1)[0],:])

        next_skin_tone = 0
        next_gender = 0
        next_age = 0
        next_pred = 0

        final_preds = [np.array([[None]*10]*fnf_preds.size),np.array([[None]*2]*fnf_preds.size),np.array([[None]*4]*fnf_preds.size)]

        for i in range(fnf_preds.size):
            if fnf_preds[i] == 0:
                skin_tone_rand = np.array([(1 if x == next_skin_tone else 0) for x in range(10)])
                gender_rand = np.array([(1 if x == next_gender else 0) for x in range(2)])
                age_rand = np.array([(1 if x == next_age else 0) for x in range(4)])

                next_skin_tone = (next_skin_tone + 1) % 10
                next_gender = (next_gender + 1) % 2
                next_age = (next_age + 1) % 4

                final_preds[0][i,:] = skin_tone_rand
                final_preds[1][i,:] = gender_rand
                final_preds[2][i,:] = age_rand
            else:
                final_preds[0][i,:] = predictions[0][next_pred]
                final_preds[1][i,:] = predictions[1][next_pred]
                final_preds[2][i,:] = predictions[2][next_pred]

                next_pred = next_pred + 1

        return final_preds

def train_validation_split(X, Y):
    nntrain = [int(0.7 * X[i].shape[0]) for i in range(len(X))]
    np.random.seed(42)
    indices = [np.random.permutation(X[i].shape[0]) for i in range(len(X))]
    idxes = [[indices[i][:nntrain[i]], indices[i][nntrain[i]:]] for i in range(len(X))]

    return [X[i][idxes[i][0], :] for i in range(len(X))], [X[i][idxes[i][1], :] for i in range(len(X))], [
        Y[i][idxes[i][0], :] for i in range(len(X))], [Y[i][idxes[i][1], :] for i in range(len(X))]

def train_validation_split_single(X, Y):
    nntrain = int(0.7 * X.shape[0])
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    idxes = [indices[:nntrain], indices[nntrain:]]

    return X[idxes[0], :], X[idxes[1], :], Y[idxes[0], :], Y[idxes[1], :]



def train_facial_categorical_model(facial_categorizer):
    def fit_model(i, model):
        print('CHANGE ME')
        model.fit(i, epochs=35, batch_size=32, save=True, save_location='./models/', verbose=True)

    threads = [None] * 3
    for i in range(3):
        threads[i] = Thread(target=fit_model, args=(i, facial_categorizer))
        threads[i].start()
    for i in range(3):
        threads[i].join()

def initializeModel():
    res_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=K.Input(shape=[64, 64, 3]))

    # freeze all but the last layer
    # for layer in res_model.layers[:143]:
    #     layer.trainable = False
    model = K.models.Sequential()
    model.add(res_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    return model