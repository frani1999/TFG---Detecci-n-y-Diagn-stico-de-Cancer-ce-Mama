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
