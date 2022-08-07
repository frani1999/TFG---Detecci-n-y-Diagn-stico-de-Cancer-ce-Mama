# TFG - Detección y Diagnóstico de Cáncer de Mama

Este TFG tiene como objetivo el estudio de distintas técnicas de detección y diagnóstico de Cáncer de Mama mediante el uso de algoritmos de Inteligencia Artificial basados en Machine Learning y Deep Learning. Como fuente de datos se utilizan imágenes mamográficas etiquetadas, tratadas de diversas formas dependiendo de la técnica aplicada.

Tras el estudio de estas técnicas se procede a realizar una comparación de los resultados para posteriormente elegir la mejor combinación de técnicas para integrar en un CADx (Computer Aided System for diagnosis) o CADe (Computer Aided System for Detection).

# Organización del Repositorio

En este repositorio se encuentran tres carpetas: Carpeta de Datos, Carpeta de Código y Carpeta de Resultados.

En la carpeta de datos, se recogen varios archivos tanto en formato csv como xls, utilizados para asignar etiquetas a imágenes o para extraer cierto subconjunto de imágenes. Las imágenes del dataset se pueden descargar aquí: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

En la carpeta de resultados se encuentran los resultados obtenidos en cada experimento tanto en la parte de detección como en la de diagnóstico. Son muy variados. Podrá ver diferentes archivos, desde gráficas hasta archivos .txt donde visualizar los logs de ejecución de cada prueba. En concreto para la parte de diagnóstico, hay una carpeta de resultados que nos devuelve YOLOv5 una vez terminada la fase de entrenamiento, y otra tras ejecutar la fase de prueba o testeo.

En la carpeta código se recoge el código python desarrollado para cada prueba realizada. Dentro de esta, encontramos:
  -una carpeta llamada "Detección", que contiene los códigos asociados a la parte de detección del proyecto.
  -una carpeta llamada "Diagnóstico", que contiene los códigos asociados a la parte de diagnóstico del proyecto.
  -una carpeta llamada "Otros", que contiene otros códigos que poseen cierta relevancia en la realización del proyecto. Dentro de esa carpeta se encuentra una carpeta que se llama MIAS. Esta carpeta contiene los códigos que se utilizaron al principio del proyecto para familiarizarse con las funciones de python y también códigos obsoletos que se usaron en técnicas descartadas para el proyecto. Todos estos códigos se probaron con la base de dats MIAS, que es mas pequeña que la base de datos CBIS-DDSM.

# Resultados obtenidos

## Resultados Detección

CNN diseñada:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/My%20CNN%20and%20VGG16/Detection%20Result%20-%20My%20CNN.png)

Red VGG16:
![Alt text](https://github.com/frani1999/TFG---Detecci-n-y-Diagn-stico-de-Cancer-ce-Mama/blob/main/Resultados/Detecci%C3%B3n/My%20CNN%20and%20VGG16/Detection%20Result%20-%20VGG16.png)

### Detección con YOLOv5

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

## Reusltados de Diagnóstico utilizando las técnicas de Machine Learning

Sensibilidad:
| Técnica utilizada | feature extraction(%) | feature extraction bounded(%) |
| ------------- | ------------- | ------------- |
| KNN |  |  |
| Logistic Regression |  |  |
| svc |  |  |
| Random Forest  |  |  |
| Decision Tree Classifier |  |  |
| GaussianNB |  |  |

Especificidad:
| Técnica utilizada | feature extraction(%) | feature extraction bounded(%) |
| ------------- | ------------- | ------------- |
| KNN |  |  |
| Logistic Regression |  |  |
| svc |  |  |
| Random Forest  |  |  |
| Decision Tree Classifier |  |  |
| GaussianNB |  |  |
