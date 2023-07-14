import cv2                              #Libreria de OpenCV, para dibujar rectangulos, cambiar colores etc.
import face_recognition                 #Libreria para el Reconocimiento Facial
import os                               #Libreria para iterar sobre los directorios

KNOWN_IDENTITIES = "Known_Identities"
UNKNOWN_IDENTITIES = "Unknown_Identities"
TOLERANCE = 0.6

known_names = []                #Aqui guardamos los nombres de los archivos conocidos
known_encodings = []            #Aqui guardamos los encodings de las caras que conocemos

for file in os.listdir(KNOWN_IDENTITIES):
    if file != ".DS_Store":
        image = face_recognition.load_image_file(f"{KNOWN_IDENTITIES}//{file}")
        encoding = face_recognition.face_encodings(image)[0]        #Posicion 0 para obtener solo una cara, ya que face_encodings toma las coordenadas suficientes para todas las caras, pero en la carpeta de KNOWN_IDENTITIES se debe tener solo una persona por imagen (que la identifique). Por eso tomamos solo la posicion 0
        known_names.append(file)
        known_encodings.append(encoding)

for file in os.listdir(UNKNOWN_IDENTITIES):
    if file != ".DS_Store":
        image = face_recognition.load_image_file(f"{UNKNOWN_IDENTITIES}//{file}")
        encodings = face_recognition.face_encodings(image)          #Sacamos lista de todas las caras codificadas que hemos identificado
        locations = face_recognition.face_locations(image)          #Sacamos lista de las localizaciones(coordenadas) de las caras identificadas
        for encoding, location in zip(encodings, locations):        #Bucle que itera sobre la lista de localizaciones y codificaciones de caras simultaneamente (como tienen el mismo numero de elementos, no nos tenemos preocupar por el numero de iteraciones)
            matches = face_recognition.compare_faces(known_encodings, encoding, TOLERANCE)  #Array de la misma longitud que la lista de identidades conocidas con true o false en las mismas posiciones de acuerdo a si la identidad conocida pertenece a la imagen que estamos analizando
            if True in matches:                                     #Si hay algun True
                match_name = known_names[matches.index(True)]       #Se toma el indice de la primera posicion que contenga un true y se obtiene el nombre correspondiente a esa posicion en la lista de nombres de identidades conocidas
                print(f"Match found: {match_name}")
                top_left = (location[3], location[0])
                bottom_righ = (location[1], location[2])
                cv2.rectangle(image, top_left, bottom_righ, (0, 255, 0), 16)
                cv2.putText(image, match_name, (location[3]+100, location[2]+200), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 255, 255), 16)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)              #Esta linea sirve para cambiar el color de la imagen a mostrar. Se puede poner aqui o mas arriba
        cv2.imshow(file, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#https://www.youtube.com/watch?v=535acCxjHCI
#https://face-recognition.readthedocs.io/en/latest/face_recognition.html
