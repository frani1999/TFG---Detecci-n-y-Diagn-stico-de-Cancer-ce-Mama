# TFG - Detección y Diagnóstico de Cáncer de Mama

Este TFG tiene como objetivo el estudio de distintas técnicas de detección y diagnóstico de Cáncer de Mama mediante el uso de algoritmos de Inteligencia Artificial basados en **Machine Learning** y **Deep Learning**. Como fuente de datos se utilizan imágenes mamográficas etiquetadas, tratadas de diversas formas dependiendo de la técnica aplicada.

Tras el estudio de estas técnicas se procede a realizar una comparación de los resultados para posteriormente elegir la mejor combinación de técnicas para integrar en un CADx o CADe.

# Organización del Repositorio

En este repositorio se encuentran tres carpetas: Carpeta de Datos, Carpeta de Código y Carpeta de Resultados.

En la carpeta de datos, se recogen varios archivos tanto en formato csv como xls, utilizados para asignar etiquetas a imágenes o para extraer cierto subconjunto de imágenes. Las imágenes del dataset se pueden descargar aquí: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

En la carpeta de resultados se encuentran los resultados obtenidos en cada experimento tanto en la parte de detección como en la de diagnóstico. Son muy variados. Podrá ver diferentes archivos, desde gráficas hasta archivos .txt donde visualizar los logs de ejecución de cada prueba. En concreto para la parte de diagnóstico, hay una carpeta de resultados que nos devuelve YOLOv5 una vez terminada la fase de entrenamiento, y otra tras ejecutar la fase de prueba o testeo.

En la carpeta código se recoge el código python desarrollado para cada prueba realizada. Dentro de esta, encontramos:
  -una carpeta llamada *"Detección"*, que contiene los códigos asociados a la parte de detección del proyecto.
  -una carpeta llamada *"Diagnóstico"*, que contiene los códigos asociados a la parte de diagnóstico del proyecto.
  -una carpeta llamada *"Otros"*, que contiene otros códigos que poseen cierta relevancia en la realización del proyecto. Dentro de esa carpeta se encuentra una carpeta que se llama **MIAS**. Esta carpeta contiene los códigos que se utilizaron al principio del proyecto para familiarizarse con las funciones de python y también códigos obsoletos que se usaron en técnicas descartadas para el proyecto. Todos estos códigos se probaron con la base de datos **MIAS**, que es mas pequeña que la base de datos **CBIS-DDSM**.

# Técnicas empleadas

Se han empleado las siguientes técnicas de **Deep Learning**:

- Se ha diseño de Red Neuronal Convolucional con la siguiente arquitectura:

![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resources/Mi%20CNN.png)

- Red VGG16, que posee la siguiente arquitectura:

![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resources/Red%20VGG16.png)

- Reentrenamiento de **YOLOv5**: Se ha conseguido adaptar el algoritmo de detección de objetos en imágenes a detección de tumores en imágenes mamográficas. Para ello, se han etiquetado manualmente y en el formato de **YOLOv5** un subconjunto de mamografías extraido de CBIS-DDSM, para posteriormente realizar el entrenamiento.
  - Enlace al repositorio de **YOLOv5**: https://github.com/ultralytics/yolov5
  - Tutorial de reentrenamiento de **YOLOv5**: https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb#scrollTo=X7yAi9hd-T4B

Por otro lado se han utilizado las siguientes técnicas de **Machine Learning**:

- *K-Nearest Neighbors (KNN)*
- *Logistic Regression*
- *Support Vector Machine (SVM)*
- *Random Forest*
- *Decision Tree Classifier*
- *Naive Bayes Classifier (GaussianNB)*

# Resultados obtenidos

## Resultados Detección

CNN diseñada:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/My%20CNN%20and%20VGG16/Detection%20Result%20-%20My%20CNN.png)

Red VGG16:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/My%20CNN%20and%20VGG16/Detection%20Result%20-%20VGG16.png)

### Detección con **YOLOv5**

Matriz de confusión:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/YOLOv5/Train/confusion_matrix.png)

Métricas:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/YOLOv5/Train/results.png)

Algunos resultados de ejemplo:

![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/YOLOv5/Test/test1.jpg)
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/YOLOv5/Test/test11.jpg)
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/YOLOv5/Test/test19.jpg)
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/YOLOv5/Test/test22.jpg)
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/YOLOv5/Test/test33.jpg)
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/YOLOv5/Test/test5.jpg)

## Resultados Diagnóstico

Calcificaciones CNN diseñada:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Diagn%C3%B3stico/My%20CNN%20and%20VGG16/Calcificaciones/Calc%20Diagnosis%20My%20CNN.png)

Calcificaciones Red VGG16:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Diagn%C3%B3stico/My%20CNN%20and%20VGG16/Calcificaciones/Calc%20Diagnosis%20VGG16.png)

Masas CNN diseñada:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Diagn%C3%B3stico/My%20CNN%20and%20VGG16/Masas/Diagnosis_My_CNN_masses.png)

Masas Red VGG16:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Diagn%C3%B3stico/My%20CNN%20and%20VGG16/Masas/Diagnosis_VGG16_masses.png)

## Resultados de Diagnóstico utilizando las técnicas de Machine Learning

Fórmula Sensibilidad:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resources/CodeCogsEqn%20(1).png)

Fórmula Especificidad:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resources/CodeCogsEqn%20(2).png)

Sensibilidad:
| Técnica utilizada | feature extraction(%) | feature extraction bounded(%) |
| ------------- | ------------- | ------------- |
| KNN | 36.57 | 37.09 |
| Logistic Regression | 54.24 | 53.56 |
| svc | 55.88 | 55.88 |
| Random Forest  | 44.80 | 45.36 |
| Decision Tree Classifier | 45.60 | 39.66 |
| GaussianNB | 57.68 | 57.39 |

Especificidad:
| Técnica utilizada | feature extraction(%) | feature extraction bounded(%) |
| ------------- | ------------- | ------------- |
| KNN | 26.04 | 25.86 |
| Logistic Regression | 33.96 | 26.83 |
| svc | 20.21 | 20.21 |
| Random Forest  | 28.29 | 30.13 |
| Decision Tree Classifier | 30.86 | 26.74 |
| GaussianNB | 18.44 | 18.33 |

Como en ambos casos se realiza una clasificación binaria, se puede coger lo contrario:

Sensibilidad:
| Técnica utilizada | feature extraction(%) | feature extraction bounded(%) |
| ------------- | ------------- | ------------- |
| KNN | 63.43 | 62.91 |
| Logistic Regression | 45.76 | 46.44 |
| svc | 44.12 | 44.12 |
| Random Forest  | 55.20 |  54.64|
| Decision Tree Classifier | 54.40 | 60.34 |
| GaussianNB | 42.32 | 42.61 |

Especificidad:
| Técnica utilizada | feature extraction(%) | feature extraction bounded(%) |
| ------------- | ------------- | ------------- |
| KNN | 73.96 | 74.14 |
| Logistic Regression | 66.04 | 73.17 |
| svc | 79.79 | 79.79 |
| Random Forest  | 71.71 | 69.87 |
| Decision Tree Classifier | 69.14 | 73.26 |
| GaussianNB | 81.56 | 81.67 |
