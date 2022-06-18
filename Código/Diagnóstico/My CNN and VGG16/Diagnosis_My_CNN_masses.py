import cv2
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import binary_fill_holes
import skimage
import datetime

class CBISDDSM():
    def __init__(self, data_path):

        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.severity = self.data['severity']
        self.labels = self.severity.replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
        self.cropped_image_name = self.data['image_folder'] + "_" + self.data['image_name']

    def read_images(self):
        self.images = []
        for i in range(len(self.cropped_image_name)):
            #print(self.cropped_image_name[i])
            img = cv2.imread("T:/fbr/CBIS-DDSM/preprocessing_masses/" + self.cropped_image_name[i], 0)

            #Resize for addapt to input of model:
            img_resize = cv2.resize(img, (224, 224))

            self.images.append(img_resize)

    def encode_labels(self):
        lb = LabelEncoder()
        self.labels_encoded = lb.fit_transform(self.labels)

    def generate_rotated_images(self):
        x = []
        y = []
        i = 0  # Index for labels_encoded
        for image in self.images:
            rows, cols = image.shape[:2]
            for angle in range(0, 360, 90):
                # "Rotación":
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)  # Rotate 0 degree
                img_rotated = cv2.warpAffine(image, M, (224, 224))
                x.append(img_rotated)
                y.append(self.labels_encoded[i])
            i += 1
        return x, y

#Path to train data:
path_train = "image_crop_masses_train.csv"
#Path to test data:
path_test = "image_crop_masses_test.csv"

print("TRAIN")
print("Leyendo imágenes del conjunto de train...")

data_train = CBISDDSM(path_train)
data_train.read_images()
data_train.encode_labels()
x_train, y_train = data_train.generate_rotated_images()
x_train = np.array(x_train)
y_train = np.array(y_train)

print("TEST")
print("Leyendo imágenes del conjunto de test...")

data_test = CBISDDSM(path_test)
data_test.read_images()
x_test = data_test.images
data_test.encode_labels()
y_test = data_test.labels_encoded
x_test = np.array(x_test)
y_test = np.array(y_test)

# Reshape to get format [BS, xx, yy, F]:
x_train = np.reshape(x_train, (x_train.shape[0],224, 224, 1)) # 1 for gray scale
x_test = np.reshape(x_test, (x_test.shape[0],224, 224,1))

# See values of each type:
print("Train: ")
unique_train, counts_train = np.unique(y_train, return_counts=True)
result_train = np.column_stack((unique_train, counts_train))
print (result_train)
print("Test: ")
unique_test, counts_test = np.unique(y_test, return_counts=True)
result_test = np.column_stack((unique_test, counts_test))
print (result_test)

def convolutional_model_1():
    model = Sequential()
    model.add(Conv2D(56, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 1))) #32
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(112, kernel_size=(3, 3), activation='relu')) #64
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(112, kernel_size=(3, 3), activation='relu')) #64
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(56, kernel_size=(3, 3), activation='relu')) #32
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(56, kernel_size=(3, 3), activation='relu')) #32
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(56, kernel_size=(3, 3), activation='relu')) #32
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu')) #128
    model.add(Dropout(0.6))
    model.add(Dense(2048, activation='relu')) #64
    model.add(Dropout(0.6))
    model.add(Dense(2048, activation='relu')) #32
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid')) #1
    return model

# Create model:
Model = convolutional_model_1()

model_check_point = ModelCheckpoint(filepath ='./', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

# Define optimizer, or use Adam:
opt = tf.keras.optimizers.Adam(learning_rate = 0.00001) # momentum = 0.9, decay = 0.0
Model.compile(optimizer= opt, # opt 'adam'
              loss='mse', #binary_crossentropy
              metrics=['accuracy'])
Model.summary()

hist = Model.fit(x_train, y_train, validation_split=0.2, epochs=60, batch_size=32) #callbacks=[early_stop]

loss_value , accuracy = Model.evaluate(x_test, y_test)
print('Test_loss_value = ' +str(loss_value))
print('test_accuracy = ' + str(accuracy))
#----------------------------------------------VISUALIZE RESULTS--------------------------------------------------------
def Visualize_Result(acc, val_acc, loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(nrows=1,
                                   ncols=2,
                                   figsize=(15, 6),
                                   sharex=True)

    plot1 = ax1.plot(range(0, len(acc)),
                     acc,
                     label='accuracy')

    plot2 = ax1.plot(range(0, len(val_acc)),
                     val_acc,
                     label='val_accuracy')

    ax1.set(title='Accuracy And Val Accuracy progress',
            xlabel='epoch',
            ylabel='accuracy/ validation accuracy')

    ax1.legend()

    plot3 = ax2.plot(range(0, len(loss)),
                     loss,
                     label='loss')

    plot4 = ax2.plot(range(0, len(val_loss)),
                     val_loss,
                     label='val_loss')

    ax2.set(title='Loss And Val loss progress',
            xlabel='epoch',
            ylabel='loss/ validation loss')

    ax2.legend()

    fig.suptitle('Result Of Model', fontsize=20, fontweight='bold')
    fig.savefig('Accuracy_Loss_figure_' + str(datetime.datetime.now().strftime('%Y-%m-%d-%Hh%Mm%Ss')) + '.png')
    plt.tight_layout()
    plt.show()

visualize_result = Visualize_Result(hist.history['accuracy'], hist.history['val_accuracy'], hist.history['loss'], hist.history['val_loss'])




