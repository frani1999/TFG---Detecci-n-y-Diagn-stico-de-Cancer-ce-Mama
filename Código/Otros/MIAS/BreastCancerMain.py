import pandas as pd
import numpy as np
from MIASDatabase import MIASDatabase
from Models import convolutional_model_1, convolutional_model_2
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import datetime

# Detection: python BreastCancerMain.py 0 1
# Diagnosis: python BreastCancerMain.py 1 1

# check input arguments:
if len(sys.argv) < 3:
    print("Use: python BreastCancerMain.py <case> <process type>")
    print("Possible cases: ")
    print("0: Detection")
    print("1: Diagnosis")
    print("Possible process type: ")
    print("1: Rotate images")
    print("2: Feature images with tissue and rotate")
    sys.exit(1)
# Read arguments:
number_case = int(sys.argv[1])
process_type = int(sys.argv[2])
if (process_type != 1)and(process_type != 2):
    raise ValueError(process_type, " is not a valid process type. Choose between 1 (rotate images) or 2 (feature images with tissue and rotate).")

# Path to images:
#path_to_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/croped_images/"
#path_to_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/anotations_removed/"
path_to_images = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/anotations_removed_preprocessing_stack/"

if number_case == 0: # Detection
    # Paths to csv detection files:
    path_to_labels_train = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/info/detection/MIAS-detection train.csv"
    path_to_labels_test = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/info/detection/MIAS-detection test.csv"
    case = "detection"
elif number_case == 1: # Diagnosis
    # Path to csv diagnosis files:
    path_to_labels_train = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/info/diagnosis/MIAS-diagnosis train.csv"
    path_to_labels_test = "C:/Users/RDuser-E1/Desktop/TFG/all-mias/info/diagnosis/MIAS-diagnosis test.csv"
    case = "diagnosis"
else:
    raise ValueError(number_case + " is not a valid case. Choose between 0(detection) or 1(diagnosis).")

# Create MIASDatabase objects for each combination of mode and case:
MIAS_train = MIASDatabase(images_path = path_to_images, labels_path= path_to_labels_train, case= case)
MIAS_test = MIASDatabase(images_path = path_to_images, labels_path= path_to_labels_test, case= case)

'''
labels_train = MIAS_train.labels
labels_test = MIAS_test.labels
print(labels_train)
print(labels_test)
'''

# Do the data processing for train set:
x_train, y_train = MIAS_train.data_process(mode="train", process_type=process_type, rotation_step=10)
# Read and Store the images of test:
x_test, y_test = MIAS_test.data_process(mode="test", process_type=process_type, rotation_step=10)
# Reshape to get format [BS, xx, yy, F]:
x_train = np.reshape(x_train, (x_train.shape[0],224, 224, 1)) # 1 for gray scale
x_test = np.reshape(x_test, (x_test.shape[0],224, 224,1))

print("Type: ", type(x_train), ", Shape: ", x_train.shape)
print("Type: ", type(y_train), ", Shape: ", y_train.shape)
print("Type: ", type(x_test), ", Shape: ", x_test.shape)
print("Type: ", type(y_test), ", Shape: ", y_test.shape)

# See values of each type:
print("Train: ")
unique_train, counts_train = np.unique(y_train, return_counts=True)
result_train = np.column_stack((unique_train, counts_train))
print (result_train)
print("Test: ")
unique_test, counts_test = np.unique(y_test, return_counts=True)
result_test = np.column_stack((unique_test, counts_test))
print (result_test)

#------------------------------------------------------MODEL------------------------------------------------------------
# Create model:
Model = convolutional_model_1()

# Callbacks:
# Frequency to save weights of the model:
model_check_point = ModelCheckpoint(filepath ='./', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

# Define optimizer, or use Adam:
opt = tf.keras.optimizers.Adam(learning_rate = 0.01) # momentum = 0.9, decay = 0.0
Model.compile(optimizer= opt, # opt 'adam'
              loss='binary_crossentropy',
              metrics=['accuracy'])
Model.summary()
#-----------------------------------------------------TRAIN-------------------------------------------------------------
hist = Model.fit(x_train, y_train, validation_split=0.2, epochs=15, batch_size=128) #callbacks=[early_stop]

#-----------------------------------------------------TEST--------------------------------------------------------------
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