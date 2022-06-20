import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

class MIASDatabase():
    def __init__(self, images_path, labels_path, case):
        '''
            Init a MIASDatabase object with the inputs parameter:
                images_path: path to the folder that contains images from MIAS Database.
                labels_path: path to csv_file that contains a csv file with labels and features of each image.
                case: detection or diagnosis, for extract diferents labels.
        '''
        self.images_path = images_path
        self.labels_path = labels_path
        self.case = case

        #Read Dataset:
        self.data = pd.read_csv(self.labels_path)

        #Extract reference of images (name of the images):
        self.reference = self.data['reference']
        self.images_name = np.array(self.reference + ".pgm")

        #Another data to use:
        self.abnormality = self.data['abnormality'] # in case of detection.
        self.severity = self.data['severity'] # in case of diagnosis.
        self.tissue = self.data['tissue'] # feature to add to our images.

        #Select  labels according to case:
        if self.case.lower() == 'detection':
            #Convert into a binary labeled data and select abnormilities like labels:
            self.labels = self.abnormality.replace(['CALC', 'CIRC', 'SPIC', 'MISC', 'ARCH', 'ASYM'], 'ANOM')
        elif self.case.lower() == 'diagnosis':
            #Select severities like labels:
            self.labels = self.severity
        else:
            raise ValueError(self.case + " is not a valid case. Choose between detection or diagnosis.")

    def data_process(self, mode,  process_type=1, rotation_step=1):
        '''
            Here, there are some possible processing functions to aply to a set.
            With this functions, we can:
                -Read images.
                - Encode labels.
                - Do data agmentation.
                - Add features to image.
            Depending of the process_type, we do differents combinations of these functions.
        '''
        # Subfunction to read images:
        def read_images(images_name, images_path):
            '''
                To read the images of an specific set.
                In our case, we use it for read test images,
                because they don´t need "data augmentation".
            '''
            images = []
            cnt_tissue = 0
            for i in range(len(images_name)):
                print(images_name[i])
                img = cv2.imread(images_path + images_name[i], 0)
                img = cv2.resize(img, (224, 224))
                # img = cv2.equalizeHist(img)
                # thresholding:
                '''
                if self.tissue[cnt_tissue] == 'F':
                    ret, img = cv2.threshold(img, 145, 255, cv2.THRESH_OTSU)
                elif self.tissue[cnt_tissue] == 'G':
                    ret, img = cv2.threshold(img, 165, 255, cv2.THRESH_OTSU)
                elif self.tissue[cnt_tissue] == 'D':
                    ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_OTSU)
                else:
                    raise ValueError(self.tissue[cnt_tissue] + " is not a valid type of tissue. Valids are F, G or D.")
                '''

                images.append(img)
                cnt_tissue += 1
            return images
        # Subfunction to encode labels:
        def encode_labels(labels):
            '''
                Type of encoding in labels:
                    Detection:
                        0: Anomaly (ANOM)
                        1: Normal (NORM)
                    Diagnosis:
                        0: Benign (B)
                        1: Malignant (M)
            '''
            lb = LabelEncoder()
            labels_encoded = lb.fit_transform(labels)
            return labels_encoded

        # Subfuntion to do "Data augmentation":
        def generate_rotated_images(images, labels_encoded, rotation_step):
            '''
                It consists of rotating each image 1º by 1º, 360 times.
                Also, encode the labels.
            '''
            x = []
            y = []
            i = 0 # Index for labels_encoded
            for image in images:
                rows, cols = image.shape
                for angle in range(0, 360, rotation_step):
                    # "Rotación":
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)  # Rotate 0 degree
                    img_rotated = cv2.warpAffine(image, M, (224, 224))
                    x.append(img_rotated)
                    y.append(labels_encoded[i])
                i += 1
            return x, y

        # Subfuntion to add feature to image:
        def add_backgroud_tissue(images, tissue_encoded):
            '''
                Put in the (0, 0) possition of the image the backgroud tissue feature.
            '''
            featured_images = []
            i = 0 # Index for tissue_encoded
            for image in images:
                image[:][0] = tissue_encoded[i]
                i += 1
                featured_images.append(image)
            return featured_images

        # Depending of the process_mode, we do one process or other:
        if process_type == 1:
            '''
                Rotate images.
            '''
            # Depending of the mode, we do an process or other:
            if mode.lower() == 'train':
                # (1) Read images:
                self.images = read_images(self.images_name, self.images_path)
                # (2) Encoding labels & feature (tissue):
                self.labels_encoded = encode_labels(self.labels)
                # (3) Rotate Images:
                x, y = generate_rotated_images(self.images, self.labels_encoded, rotation_step)
            elif mode.lower() == 'test':
                # (1) Read images:
                self.images = read_images(self.images_name, self.images_path)
                # (2) Encoding labels & feature (tissue):
                self.labels_encoded = encode_labels(self.labels)
                x, y = generate_rotated_images(self.images, self.labels_encoded, rotation_step)
            else:
                raise ValueError(mode + " is not a valid mode. Choose between train or test.")
        elif process_type == 2:
            '''
                Add tissue feature to the images and rotate.
            '''
            # Depending of the mode, we do an process or other:
            if mode.lower() == 'train':
                # (1) Read images:
                self.images = read_images(self.images_name, self.images_path)
                # (2) Encoding labels & feature (tissue):
                self.labels_encoded = encode_labels(self.labels)
                self.tissue_encoded = encode_labels(self.tissue)

                print("tissue_encoded: ")
                unique_tissue, counts_tissue = np.unique(self.tissue_encoded, return_counts=True)
                result_tissue = np.column_stack((unique_tissue, counts_tissue))
                print(result_tissue)

                # (3) Featured images with backgroud tissue:
                self.featured_images = add_backgroud_tissue(self.images, self.tissue_encoded)
                # (4) Data augmentation:
                x, y = generate_rotated_images(self.featured_images, self.labels_encoded, rotation_step)
            elif mode.lower() == 'test':
                # (1) Read images:
                self.images = read_images(self.images_name, self.images_path)
                # (2) Encoding labels & feature (tissue):
                self.labels_encoded = encode_labels(self.labels)
                y = self.labels_encoded
                self.tissue_encoded = encode_labels(self.tissue)

                print("tissue_encoded: ")
                unique_tissue, counts_tissue = np.unique(self.tissue_encoded, return_counts=True)
                result_tissue = np.column_stack((unique_tissue, counts_tissue))
                print(result_tissue)

                # (3) Featured images with backgroud tissue:
                x = add_backgroud_tissue(self.images, self.tissue_encoded)
            else:
                raise ValueError(mode + " is not a valid mode. Choose between train or test.")
        else:
            raise ValueError(process_type, " is not a valid process type. Choose between 1 (rotate images) or 2 (feature images with tissue and rotate).")
        x = np.array(x)
        y = np.array(y)
        return x, y