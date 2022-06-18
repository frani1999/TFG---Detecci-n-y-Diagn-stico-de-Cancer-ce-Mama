import pandas as pd
import cv2
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import models, layers
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import tensorflow as tf
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
            img = cv2.imread("T:/fbr/CBIS-DDSM/preprocessing_masses/" + self.cropped_image_name[i])

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

# See values of each type:
print("Train: ")
unique_train, counts_train = np.unique(y_train, return_counts=True)
result_train = np.column_stack((unique_train, counts_train))
print (result_train)
print("Test: ")
unique_test, counts_test = np.unique(y_test, return_counts=True)
result_test = np.column_stack((unique_test, counts_test))
print (result_test)


def create_vgg16(verbose=False, fc_size=1536, dropout=0.6): #256

    vgg16_base = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(224, 224, 3))

    vgg16 = models.Sequential()
    vgg16.add(vgg16_base)

    vgg16.add(layers.Flatten())
    if dropout is not None:
        vgg16.add(layers.Dropout(dropout))
    #vgg16.add(BatchNormalization())
    vgg16.add(layers.Dense(fc_size, activation='relu')) #, kernel_regularizer=tf.keras.regularizers.L2(0.005)
    vgg16.add(layers.Dropout(0.6))
    #vgg16.add(BatchNormalization())
    vgg16.add(layers.Dense(fc_size/2, activation='relu')) #, kernel_regularizer=tf.keras.regularizers.L2(0.005)
    vgg16.add(layers.Dropout(0.4))
    #vgg16.add(BatchNormalization())
    vgg16.add(layers.Dense(1, activation='sigmoid'))

    # Freeze the convolutional base
    vgg16_base.trainable = False

    if verbose:
        vgg16_base.summary()
        vgg16.summary()

    return vgg16

# Create model:
Model = create_vgg16()

# Callbacks:
# Frequency to save weights of the model:
model_check_point = ModelCheckpoint(filepath ='./', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

# Define optimizer, or use Adam: #0.000001
opt = tf.keras.optimizers.Adam(learning_rate = 0.000001) # momentum = 0.9, decay = 0.0
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




