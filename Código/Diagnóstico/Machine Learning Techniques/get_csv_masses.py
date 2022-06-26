import cv2
import matplotlib.pyplot as plt
import skimage
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.measure import regionprops
import math
import pandas as pd

#Path to train data:
path_train = "C:/Users/RDuser-E1/Desktop/TFG/Diagnosis/masses/image_crop_masses_train.csv"
#Path to test data:
path_test = "C:/Users/RDuser-E1/Desktop/TFG/Diagnosis/masses/image_crop_masses_test.csv"

#Train:
print("\n\n----------TRAIN------------\n\n")

#Generate output dataframe:
train_output = pd.DataFrame()
train_output['image_folder'] = []
train_output['image_name'] = []
train_output['area'] = []
train_output['bounding box area'] = []
train_output['convex area'] = []
train_output['exentricity'] = []
train_output['equivalent diameter'] = []
train_output['extension'] = []
train_output['feret diameter'] = []
train_output['major axis length'] = []
train_output['minor axis length'] = []
train_output['orientation'] = []
train_output['perimeter'] = []
train_output['solidity'] = []
train_output['compactness'] = []
train_output['severity'] = []

#Read train data:
train_data = pd.read_csv(path_train)
cropped_image_folder = train_data['image_folder']
cropped_image_name = train_data['image_name']
severity = train_data['severity']

for i in range(len(cropped_image_folder)):
    print("T:/fbr/CBIS-DDSM/jpeg/" + cropped_image_folder[i] + "/" + cropped_image_name[i])
    img = cv2.imread("T:/fbr/CBIS-DDSM/jpeg/" + cropped_image_folder[i] + "/" + cropped_image_name[i], 0)
    img = img/255
    '''
    fig = plt.subplot()
    plt.imshow(img, cmap='gray')
    plt.title("Imagen original Normalizada!")
    plt.show()
    '''
    #Aplicamos filtro gausiano a la imagen:
    img_gaus = (cv2.GaussianBlur(img, (5, 5), 0) * 255).astype('uint8')
    '''
    fig = plt.subplot()
    plt.imshow(img_gaus, cmap='gray')
    plt.title("Filtro Gausiano!")
    plt.show()
    '''
    #Umbralizamos la imagen, con método OTSU:
    t, mask = cv2.threshold(img_gaus, 0, 255, cv2.THRESH_OTSU)
    mask_norm = ((255 - mask)/255) == 0
    '''
    fig = plt.subplot()
    plt.imshow(mask_norm, cmap='gray')
    plt.title("Máscara Otsu t = " + str(t))
    plt.show()
    '''
    #Eliminamos artefactos:
    img_removed = skimage.morphology.remove_small_objects(mask_norm, min_size=64)
    '''
    fig = plt.subplot()
    plt.imshow(img_removed, cmap='gray')
    plt.title("Eliminados los artefactos!")
    plt.show()
    '''
    #Rellenar los objetos de los lúmenes:
    mask2 = binary_fill_holes(img_removed)
    '''
    fig = plt.subplot()
    plt.imshow(mask2, cmap='gray')
    plt.title("Rellenados los objetos de los lúmenes!")
    plt.show()
    '''
    #-----------------------------------------------------------------------------------------------------------------------
    #Detectar y dibujar los contornos de los lúmenes sobre la imagen original:
    #Detección de bordes con canny:
    mask2_255 = (mask2 * 255).astype('uint8')
    canny = cv2.Canny(mask2_255, 255/3, 255) #minValue y maxValue. Se recomienda que sea 3 veces mayor minValue que maxValue.
    kernel = np.ones((2, 2), 'uint8') #4
    canny = cv2.dilate(canny, kernel, iterations=2)
    '''
    plt.imshow(canny, cmap='gray')
    plt.title("Bordes detectados con Canny!")
    plt.show()
    '''
    index = np.array(np.where(canny==255))
    #Pintamos sobre la imagen original:
    img_c = img.copy()
    img_c[index[0, :], index[1, :]] = 0
    '''
    plt.imshow(img_c, cmap='gray')
    plt.title("imagen original con lúmenes identificados!")
    plt.show()
    '''

    #Identificar y cropear el lumen más grande:

    lab, num = label(mask2, return_num=True)
    v, c = np.unique(lab, return_counts=True)

    #Eliminar primera componente(fondo):
    c = c[1:]
    v = v[1:]

    #Representamos lumen más grande:
    big_lumen = lab==v[c.argmax()]
    '''
    plt.imshow(big_lumen, cmap='gray')
    plt.title("Lumen más grande!")
    plt.show()
    '''
    #Cropping:
    index = np.where(big_lumen)
    cropped_lumen = big_lumen[index[0].min():index[0].max()+1, index[1].min():index[1].max()+1]
    '''
    plt.imshow(cropped_lumen, cmap='gray')
    plt.title("Lumen más grande cropeado!")
    plt.show()
    '''
    #Extraer 13 características geométricas para caracterizar el lumen recortado:
    #Calcular las siguientes características del crop y redondear hasta el cuarto decimal:

    props = regionprops(cropped_lumen.astype(np.uint8))
    # 1) Área
    area = np.round(props[0].area,4)
    print("Área:", area)
    # 2) Área de la bounding box
    area_bb = np.round(props[0].bbox_area,4)
    print("Área de la bounding box:", area_bb)
    # 3) Área convexa
    area_conv = np.round(props[0].convex_area,4)
    print("Área convexa:", area_conv)
    # 4) Exentricidad
    eccentricity = np.round(props[0].eccentricity,4)
    print("Exentricidad:", eccentricity)
    # 5) Diámetro equivalente
    equivalent_diameter = np.round(props[0].equivalent_diameter,4)
    print("Diametro equivalente:", equivalent_diameter)
    # 6) Extensión
    extent = np.round(props[0].extent,4)
    print("Extensión:", extent)
    # 7) Diámetro Feret
    feret_diameter_max = np.round(props[0].feret_diameter_max,4)
    print("Diámetro Feret:", feret_diameter_max)
    # 8) Longitud del eje mayor
    major_axis_length = np.round(props[0].major_axis_length,4)
    print("Longitud del eje mayor:", major_axis_length)
    # 9) Longitud del eje menor
    minor_axis_length = np.round(props[0].minor_axis_length,4)
    print("Longitud del eje menor:", minor_axis_length)
    # 10) Orientación
    orientation = np.round(props[0].orientation,4)
    print("Orientación:", orientation)
    # 11) Perímetro
    perimeter = np.round(props[0].perimeter,4)
    print("Perímetro:", perimeter)
    # 12) Solidez
    solidity = np.round(props[0].solidity,4)
    print("Solidez:", solidity)
    # 13) Compacidad
    compactness = np.round(4*math.pi*props[0].area/props[0].perimeter**2, 4)
    print("Compacidad:", compactness)

    train_output = train_output.append(pd.Series([cropped_image_folder[i], cropped_image_name[i], area, area_bb,
                                                  area_conv, eccentricity, equivalent_diameter, extent, feret_diameter_max,
                                                  major_axis_length, minor_axis_length, orientation, perimeter, solidity,
                                                  compactness, severity[i]], index=['image_folder', 'image_name', 'area', 'bounding box area',
                                                  'convex area', 'exentricity', 'equivalent diameter', 'extension', 'feret diameter',
                                                  'major axis length','minor axis length', 'orientation', 'perimeter', 'solidity',
                                                  'compactness', 'severity']), ignore_index=True)
    print(cropped_image_folder[i], "|", cropped_image_name[i], "|", area, "|", area_bb, "|",
                                                  area_conv, "|", eccentricity, "|", equivalent_diameter, "|", extent, "|", feret_diameter_max, "|",
                                                  major_axis_length, "|", minor_axis_length, "|", orientation, "|", perimeter, "|", solidity, "|",
                                                  compactness, "|", severity[i])

print(train_output)
train_output.to_csv("train_mass_crops_features.csv", index=False)

# Test:
print("\n\n----------TEST------------\n\n")

# Generate output dataframe:
test_output = pd.DataFrame()
test_output['image_folder'] = []
test_output['image_name'] = []
test_output['area'] = []
test_output['bounding box area'] = []
test_output['convex area'] = []
test_output['exentricity'] = []
test_output['equivalent diameter'] = []
test_output['extension'] = []
test_output['feret diameter'] = []
test_output['major axis length'] = []
test_output['minor axis length'] = []
test_output['orientation'] = []
test_output['perimeter'] = []
test_output['solidity'] = []
test_output['compactness'] = []
test_output['severity'] = []

# Read train data:
test_data = pd.read_csv(path_test)
cropped_image_folder = test_data['image_folder']
cropped_image_name = test_data['image_name']
severity = test_data['severity']

for i in range(len(cropped_image_folder)):
    print("T:/fbr/CBIS-DDSM/jpeg/" + cropped_image_folder[i] + "/" + cropped_image_name[i])
    img = cv2.imread("T:/fbr/CBIS-DDSM/jpeg/" + cropped_image_folder[i] + "/" + cropped_image_name[i], 0)
    img = img / 255
    '''
    fig = plt.subplot()
    plt.imshow(img, cmap='gray')
    plt.title("Imagen original Normalizada!")
    plt.show()
    '''
    # Aplicamos filtro gausiano a la imagen:
    img_gaus = (cv2.GaussianBlur(img, (5, 5), 0) * 255).astype('uint8')
    '''
    fig = plt.subplot()
    plt.imshow(img_gaus, cmap='gray')
    plt.title("Filtro Gausiano!")
    plt.show()
    '''
    # Umbralizamos la imagen, con método OTSU:
    t, mask = cv2.threshold(img_gaus, 0, 255, cv2.THRESH_OTSU)
    mask_norm = ((255 - mask) / 255) == 0
    '''
    fig = plt.subplot()
    plt.imshow(mask_norm, cmap='gray')
    plt.title("Máscara Otsu t = " + str(t))
    plt.show()
    '''
    # Eliminamos artefactos:
    img_removed = skimage.morphology.remove_small_objects(mask_norm, min_size=64)
    '''
    fig = plt.subplot()
    plt.imshow(img_removed, cmap='gray')
    plt.title("Eliminados los artefactos!")
    plt.show()
    '''
    # Rellenar los objetos de los lúmenes:
    mask2 = binary_fill_holes(img_removed)
    '''
    fig = plt.subplot()
    plt.imshow(mask2, cmap='gray')
    plt.title("Rellenados los objetos de los lúmenes!")
    plt.show()
    '''
    # -----------------------------------------------------------------------------------------------------------------------
    # Detectar y dibujar los contornos de los lúmenes sobre la imagen original:
    # Detección de bordes con canny:
    mask2_255 = (mask2 * 255).astype('uint8')
    canny = cv2.Canny(mask2_255, 255 / 3,
                      255)  # minValue y maxValue. Se recomienda que sea 3 veces mayor minValue que maxValue.
    kernel = np.ones((2, 2), 'uint8')  # 4
    canny = cv2.dilate(canny, kernel, iterations=2)
    '''
    plt.imshow(canny, cmap='gray')
    plt.title("Bordes detectados con Canny!")
    plt.show()
    '''
    index = np.array(np.where(canny == 255))
    # Pintamos sobre la imagen original:
    img_c = img.copy()
    img_c[index[0, :], index[1, :]] = 0
    '''
    plt.imshow(img_c, cmap='gray')
    plt.title("imagen original con lúmenes identificados!")
    plt.show()
    '''

    # Identificar y cropear el lumen más grande:

    lab, num = label(mask2, return_num=True)
    v, c = np.unique(lab, return_counts=True)

    # Eliminar primera componente(fondo):
    c = c[1:]
    v = v[1:]

    # Representamos lumen más grande:
    big_lumen = lab == v[c.argmax()]
    '''
    plt.imshow(big_lumen, cmap='gray')
    plt.title("Lumen más grande!")
    plt.show()
    '''
    # Cropping:
    index = np.where(big_lumen)
    cropped_lumen = big_lumen[index[0].min():index[0].max() + 1, index[1].min():index[1].max() + 1]
    '''
    plt.imshow(cropped_lumen, cmap='gray')
    plt.title("Lumen más grande cropeado!")
    plt.show()
    '''
    # Extraer 13 características geométricas para caracterizar el lumen recortado:
    # Calcular las siguientes características del crop y redondear hasta el cuarto decimal:

    props = regionprops(cropped_lumen.astype(np.uint8))
    # 1) Área
    area = np.round(props[0].area, 4)
    print("Área:", area)
    # 2) Área de la bounding box
    area_bb = np.round(props[0].bbox_area, 4)
    print("Área de la bounding box:", area_bb)
    # 3) Área convexa
    area_conv = np.round(props[0].convex_area, 4)
    print("Área convexa:", area_conv)
    # 4) Exentricidad
    eccentricity = np.round(props[0].eccentricity, 4)
    print("Exentricidad:", eccentricity)
    # 5) Diámetro equivalente
    equivalent_diameter = np.round(props[0].equivalent_diameter, 4)
    print("Diametro equivalente:", equivalent_diameter)
    # 6) Extensión
    extent = np.round(props[0].extent, 4)
    print("Extensión:", extent)
    # 7) Diámetro Feret
    feret_diameter_max = np.round(props[0].feret_diameter_max, 4)
    print("Diámetro Feret:", feret_diameter_max)
    # 8) Longitud del eje mayor
    major_axis_length = np.round(props[0].major_axis_length, 4)
    print("Longitud del eje mayor:", major_axis_length)
    # 9) Longitud del eje menor
    minor_axis_length = np.round(props[0].minor_axis_length, 4)
    print("Longitud del eje menor:", minor_axis_length)
    # 10) Orientación
    orientation = np.round(props[0].orientation, 4)
    print("Orientación:", orientation)
    # 11) Perímetro
    perimeter = np.round(props[0].perimeter, 4)
    print("Perímetro:", perimeter)
    # 12) Solidez
    solidity = np.round(props[0].solidity, 4)
    print("Solidez:", solidity)
    # 13) Compacidad
    compactness = np.round(4 * math.pi * props[0].area / props[0].perimeter ** 2, 4)
    print("Compacidad:", compactness)

    test_output = test_output.append(pd.Series([cropped_image_folder[i], cropped_image_name[i], area, area_bb,
                                                  area_conv, eccentricity, equivalent_diameter, extent,
                                                  feret_diameter_max,
                                                  major_axis_length, minor_axis_length, orientation, perimeter, solidity,
                                                  compactness, severity[i]],
                                                 index=['image_folder', 'image_name', 'area', 'bounding box area',
                                                        'convex area', 'exentricity', 'equivalent diameter',
                                                        'extension', 'feret diameter',
                                                        'major axis length', 'minor axis length', 'orientation',
                                                        'perimeter', 'solidity',
                                                        'compactness', 'severity']), ignore_index=True)

    print(cropped_image_folder[i], "|", cropped_image_name[i], "|", area, "|", area_bb, "|",
                                                  area_conv, "|", eccentricity, "|", equivalent_diameter, "|", extent, "|", feret_diameter_max, "|",
                                                  major_axis_length, "|", minor_axis_length, "|", orientation, "|", perimeter, "|", solidity, "|",
                                                  compactness, "|", severity[i])

print(test_output)
test_output.to_csv("test_mass_crops_features.csv", index=False)