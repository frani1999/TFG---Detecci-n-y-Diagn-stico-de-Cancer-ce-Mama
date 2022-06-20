import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
import tensorflow as tf
"""
from keras import optimizers
from keras import losses
from sklearn import metrics
from tensorflow.keras.layers import AveragePooling2D

import random # for visualization
"""


# Primero comprobamos que podemos entrenar con nuestra GPU:
print(device_lib.list_local_devices())

#Ruta a las imágenes:
path = 'C:/Users/RDuser-E1/Desktop/TFG/all-mias/'

#--------------------------PRE-PROCESADO DE LOS DATOS-------------------------:


#Leemos los datos de nuestro csv:
info = pd.read_csv("C:/Users/RDuser-E1/Desktop/TFG/all-mias/info/MIAS.csv",sep=",")
#print(info)
# Select abnormality column:
labels = info['abnormality']
#print(labels)
# To do a binary classification, change NO NORMAL cases to ANOM:
labels = labels.replace(['CALC', 'CIRC', 'SPIC', 'MISC', 'ARCH', 'ASYM'], 'ANOM')
print(labels)
# Enconding labels:
# ANOM = 0
# NORM = 1
#lb = LabelEncoder()
#labels = lb.fit_transform(labels) # Is a numpy array

#PRE-PROCESADO DE IMÉGENES:


#Obtengo el nombre/ruta a las imágenes y la guardo en img_name. Este lo transformo a numpy aray:
img_name = []
for i in range(len(labels)):
        img_name.append(path + info.reference[i]+ '.pgm')
img_name = np.array(img_name)

#Lectura de imágenes y etiquetas y transformación de las imágenes:
img_path = []
last_label = []
for i in range(len(img_name)):
    #Leemos la imagen:
    img = cv2.imread(img_name[i], 0)
    #Le hacemos un resize
    img = cv2.resize(img, (224, 224))
    rows, cols = img.shape
    for angle in range(0, 360):
        #"Rotación":
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)  # Rotate 0 degree
        img_rotated = cv2.warpAffine(img, M, (224, 224))
        img_path.append(img_rotated)
        last_label.append(labels[i])
last_label = np.array(last_label)
print(last_label)

#División en muestras de entrenamiento y de prueba:
x_train, x_test, y_train, y_test = train_test_split(img_path, last_label, test_size = 0.25, random_state = 42)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],224, 224, 1)) # 1 for gray scale
x_test = np.reshape(x_test, (x_test.shape[0],224, 224,1))

#Creación del modelo:
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224, 224, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = create_model()
model.summary()

#definimos los callbacks:
#early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5,restore_best_weights=True, verbose=1)

check_point_filepath = './'

model_check_point = ModelCheckpoint(filepath =check_point_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='auto', save_freq='epoch')

# opt = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, decay = 0.0)
model.compile(optimizer='adam', # opt
              loss='binary_crossentropy',
              metrics=['accuracy'])
#model.summary()

hist = model.fit(x_train,
                 y_train,
                 validation_split=0.2,
                 epochs=20,
                 batch_size=128
                 ) #callbacks=[early_stop]

y_test = np.array(y_test)
loss_value , accuracy = model.evaluate(x_test, y_test)

print('Test_loss_value = ' +str(loss_value))
print('test_accuracy = ' + str(accuracy))

print(model.predict(x_test))


#-----------------------Visualización de resultados----------------------------------:

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
    fig.savefig('Accuracy_Loss_figure.png')
    plt.tight_layout()
    plt.show()


visualize_result = Visualize_Result(hist.history['accuracy'], hist.history['val_accuracy'], hist.history['loss'],
                                    hist.history['val_loss'])

y_pred=model.predict(x_test)

from sklearn.metrics import roc_curve, auc

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# roc plot for specific class
plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ANNROC')
plt.legend(loc="lower right")
plt.show()