"""

import cv2

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer un cuadro del video
    ret, frame = cap.read()

    # Mostrar el cuadro en una ventana
    cv2.imshow('Video', frame)

    # Salir del bucle si el usuario presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura y cerrar la ventana
cap.release()
cv2.destroyAllWindows()

"""

#Implementacion de un programa que reconozca caras previamente conocidas a tiempo real en video
import cv2
import face_recognition
import os

KNOWN_IDENTITIES = "Known_Identities"

nombres = []                    #Aqui guardaremos los nombres de las imagenes
identidades = []                #En esta lista guardaremos las identidades conocidas (sus encodings)

for file in os.listdir(f"{KNOWN_IDENTITIES}"):
    if file != ".DS_Store":
        image = face_recognition.load_image_file(f"{KNOWN_IDENTITIES}//{file}")
        encoding = face_recognition.face_encodings(image)[0]
        identidades.append(encoding)
        nombres.append(file)

while True:
    captura = cv2.VideoCapture(0)

    ret, frame = captura.read()
    ###

    image = frame
    encodings = face_recognition.face_encodings(image)          #Sacamos lista de todas las caras codificadas que hemos identificado
    locations = face_recognition.face_locations(image)          #Sacamos lista de las localizaciones(coordenadas) de las caras identificadas
    for encoding, location in zip(encodings, locations):        #Bucle que itera sobre la lista de localizaciones y codificaciones de caras simultaneamente (como tienen el mismo numero de elementos, no nos tenemos preocupar por el numero de iteraciones)
        matches = face_recognition.compare_faces(identidades, encoding, 0.6)  #Array de la misma longitud que la lista de identidades conocidas con true o false en las mismas posiciones de acuerdo a si la identidad conocida pertenece a la imagen que estamos analizando
        if True in matches:                                     #Si hay algun True
            match_name = nombres[matches.index(True)]       #Se toma el indice de la primera posicion que contenga un true y se obtiene el nombre correspondiente a esa posicion en la lista de nombres de identidades conocidas
            print(f"Match found: {match_name}")
            top_left = (location[3], location[0])
            bottom_righ = (location[1], location[2])
            cv2.rectangle(image, top_left, bottom_righ, (0, 255, 0), 16)
            cv2.putText(image, match_name, (location[3]+10, location[2]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    ###
    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()