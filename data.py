###SOME CODE PROVIDED AS BOILERPLATE CODE FOR EIGHT BIT BIAS BOUNTY###

import dill
import imblearn
import numpy as np
import pandas as pd
from keras.applications.efficientnet_v2 import preprocess_input
from keras.utils import load_img, img_to_array, array_to_img
from mtcnn import MTCNN
from sklearn.preprocessing import LabelBinarizer


class Data:
    def __init__(self):
        self.length = self.width = 64
        self.filename_for_image = {}
        self.X = None
        self.Y = None
        self.X_SMOTED = None
        self.Y_SMOTED = None
        self.faces = None

    def load_images(self, location):

        df = pd.read_csv(location + 'labels.csv')
        df = df[df["skin_tone"].notna()]

        cat = ['skin_tone', 'gender', 'age']
        lbs = [LabelBinarizer() for _ in range(3)]

        self.Y = []
        for i in range(3):
            lab = lbs[i].fit_transform(df[cat[i]])
            if lab.shape[1] == 1:
                self.Y.append(np.hstack((1 - lab, lab)))
            else:
                self.Y.append(lab)

        nt = df.shape[0]
        all_imgs = [load_img(location + df.iloc[i]['name'], target_size=(self.length, self.width)) for i in range(nt)]

        self.X = np.empty([nt, self.length, self.width, 3], dtype=float)
        for i in range(nt):
            self.X[i, :] = img_to_array(all_imgs[i])
        self.X = preprocess_input(self.X)
        [self.add_filename_for_image(self.X[x], df.iloc[x]['name']) for x in range(nt)]

    def add_filename_for_image(self, image, filename):
        self.filename_for_image[tuple(np.reshape(image, (1, -1))[0])] = filename

    def get_filename_for_image(self, image):
        key = tuple(np.reshape(image, (1, -1))[0])
        if key not in self.filename_for_image:
            return 'image not found'
        return self.filename_for_image[key]

    def smote_data(self, X, Y):
        sm = imblearn.over_sampling.SMOTE()

        X_r = [np.reshape(X[i], (np.shape(X[i])[0], -1)) for i in range(3)]
        Y_r = [np.argmax(Y[i], axis=1) for i in range(3)]

        oversampled = [(sm.fit_resample(X_r[i], Y_r[i].transpose())) for i in range(3)]

        self.Y_SMOTED = [np.zeros((oversampled[i][1].size, oversampled[i][1].max() + 1)) for i in range(3)]
        for i in range(3):
            self.Y_SMOTED[i][np.arange(oversampled[i][1].size), oversampled[i][1]] = 1

        self.X_SMOTED = [np.reshape(np.array(oversampled[i][0]), (-1, self.length, self.width, 3)) for i in range(3)]

        return self.X_SMOTED, self.Y_SMOTED


    def learn_faces(self, confidence):
        detector = MTCNN()

        self.faces = [indx for indx in range(self.X.shape[0]) if
                      len([x for x in detector.detect_faces(self.X[indx]) if x['confidence'] > confidence])]

        dill.dump(self.faces, open('faces_indexes.dill', 'wb'))

    def load_faces(self):
        self.faces = dill.load(open('faces_indexes.dill', 'rb'))

    def load_filters(self, skin_tone_filter_file, gender_filter_file, age_filter_file, percent):
        skin_tone_error = dill.load(open(skin_tone_filter_file, 'rb'))
        gender_error = dill.load(open(gender_filter_file, 'rb'))
        age_error = dill.load(open(age_filter_file, 'rb'))

        face_indexes = self.faces

        self.skin_tone_indices = [z[1] for z in
                             sorted([(skin_tone_error[i], i) for i in range(len(skin_tone_error)) if i in face_indexes],
                                    key=lambda x: x[0])[int(percent / 100 * len(skin_tone_error)):]]
        self.gender_indices = [z[1] for z in
                          sorted([(gender_error[i], i) for i in range(len(gender_error)) if i in face_indexes],
                                 key=lambda x: x[0])[int(percent / 100 * len(gender_error)):]]
        self.age_indices = [z[1] for z in sorted([(age_error[i], i) for i in range(len(age_error)) if i in face_indexes],
                                key=lambda x: x[0])[int(percent / 100 * len(age_error)):]]

    def get_index_from_filename(self, filename, data):
        img = preprocess_input(img_to_array(load_img(filename, target_size=(self.length, self.width))))

        data = [tuple(np.reshape(x, (1, -1))[0]) for x in data]
        img_key = tuple(np.reshape(img, (1, -1))[0])

        for i in range(len(data)):
            if data[i] == img_key:
                return i

        raise Exception('Image not found!')

    def split_faces(self):
        FNF_X = self.X
        FNF_Y = np.array([[1] if i in self.faces else [0] for i in range(self.X.shape[0])])

        FNF_X_r = np.reshape(FNF_X, (np.shape(FNF_X)[0], -1))

        sm = imblearn.over_sampling.SMOTE()

        FNF_oversampled = sm.fit_resample(FNF_X_r, FNF_Y)

        FNF_Y = np.zeros((FNF_oversampled[1].size, FNF_oversampled[1].max() + 1))
        FNF_Y[np.arange(FNF_oversampled[1].size), FNF_oversampled[1]] = 1

        FNF_X = np.reshape(np.array(FNF_oversampled[0]), (-1, self.length, self.width, 3))

        return FNF_X, FNF_Y

    def get_faces(self, influence_filter=False):
        if not influence_filter:
            return [self.X[self.faces,:] for _ in range(3)], [self.Y[i][self.faces,:] for i in range(3)]
        else:
            return [self.X[self.skin_tone_indices,:], self.X[self.gender_indices,:], self.X[self.age_indices,:]], [self.Y[0][self.skin_tone_indices,:], self.Y[1][self.gender_indices,:], self.Y[2][self.age_indices,:]]
        

def load_data_test(d):
    array_to_img(d.X[50]).show()
    print(d.get_filename_for_image(d.X[50]))


def smote_data_test(d):
    d.smote_data()
    print(len(d.X_SMOTED[0]))
    array_to_img(d.X_SMOTED[0][50]).show()
    print(d.get_filename_for_image(d.X_SMOTED[0][50]))
    print(d.Y_SMOTED[0][50])
    print(d.Y_SMOTED[1][50])
    print(d.Y_SMOTED[2][50])
    array_to_img(d.X_SMOTED[0][-1]).show()
    print(d.get_filename_for_image(d.X_SMOTED[0][-1]))

def img_index_test(d):
    faces = d.get_faces()
    # print(d.get_filename_for_image(faces[0][0][3]))
    print(d.get_index_from_filename('./train/TRAIN0012.png', faces[0][0]))

if __name__ == '__main__':
    data = Data()
    data.load_images('./train/')
    #load_data_test(data)
    #smote_data_test(data)
    data.load_faces()
    # smote_data_test(data)
    img_index_test(data)
