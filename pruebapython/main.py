import cv2                      #Libreria de OpenCV
#import matplotlib
#import numpy
import face_recognition         #Libreria para el reconocimiento facial
import os                       #Libreria para iterar sobre directorios

KNOWN_IDENTITIES = "Known_Identities"
UNKNOWN_IDENTITIES = "Unknown_Identities"
TOLERANCE = 0.6
#FRAME_THICKNESS = 3             #Tamaño en pixeles de los rectangulos rodeando a la cara
#FONT_THICKNESS = 2
#MODEL = "cnn"                   #Convolutional Neural Network

print("Loading known identites")
known_identities = []
known_names = []

for name in os.listdir(KNOWN_IDENTITIES):
    if name != ".DS_Store":
        image = face_recognition.load_image_file(f"{KNOWN_IDENTITIES}//{name}")
        encoding = face_recognition.face_encodings(image)[0]                            #Se pone el 0 para que solo tome la primera cara, en caso de que haya varias caras en la misma foto. Esto no deberia pasar, ya que las caras contenidas en KNOWN_IDENTITIES deben ser UNICAS y solo aparezca UNA PERSONA en la foto
        known_identities.append(encoding)
        known_names.append(name)

print("Processing unknown identities")
for filename in os.listdir(UNKNOWN_IDENTITIES):
    print(filename)
    if filename != ".DS_Store":
        image = face_recognition.load_image_file(f"{UNKNOWN_IDENTITIES}//{filename}")
        locations = face_recognition.face_locations(image)#, model=MODEL)                 #Localizamos y aislamos todas las caras de la imagen
        encodings = face_recognition.face_encodings(image, locations)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#        image[..., ::-1]

        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_identities, face_encoding, TOLERANCE)    #Results guarda una lista con true y false de acuerdo a si una KNOWN_IDENTITY se corresponde con alguna cara de la imagen. El tamaño de results será el mismo que el numero de imagenes que haya en KNOWN_IDENTITIES
            match = None
            if True in results:
                match = known_names[results.index(True)]                                #Esto lo podemos hacer porque suponemos que solo tenemos una unica identidad en la lista de identidades conocidas. Si no fuese asi, se tomaria como match la posicion a la que se corresponde la posicion del primer true del array results
                print(f"Match found: {match}")
                top_left = (face_location[3], face_location[0])                         #Definimos el rectangulo que rodeará la cara
                bottom_right = (face_location[1], face_location[2])
#                color = [0, 255, 0]
#                cv2.rectangle(image, top_left, bottom_right, FRAME_THICKNESS)

#                top_left = (face_location[3], face_location[2])                         #Definimos el rectangulo que rodeará la cara
#                bottom_right = (face_location[1], face_location[2]+22)
#                top, right, bottom, left = face_location
#                top_left = (top, right)
#                bottom_right = (bottom, left)
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 16) #cv2.FILLED)      #16 --> Frame Thickness
                cv2.putText(image, match, (face_location[3]+100, face_location[2]+200), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 255, 255), 16)

        cv2.imshow(filename, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#https://www.youtube.com/watch?v=535acCxjHCI
#https://face-recognition.readthedocs.io/en/latest/face_recognition.html
