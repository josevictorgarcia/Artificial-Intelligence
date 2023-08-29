from __future__ import absolute_import, division, print_function, unicode_literals		#from __future__ hace que puedas compatibilizar distintas versiones de Python (respalda funciones de distintas versiones de Python al interprete actual). Esta linea se puede borrar con algunas versiones de Python 3, ya que Python 3 incluye muchas funciones del modulo future.

import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()

logger.setLevel(logging.ERROR)


dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)		#Carga el dataset y el metadata del dataset
train_dataset, test_dataset = dataset['train'], dataset['test']					#Se copian los tests de entrenamiento y de evaluación del dataset enteros

class_names = [																	#Array con las posibles salidas del programa
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis',
    'Siete', 'Ocho', 'Nueve'
]

num_train_examples = metadata.splits['train'].num_examples						#Se extrae el numero de ejemplares del dataset, tanto para entrenamiento como para evaluación
num_test_examples = metadata.splits['test'].num_examples

#Normalizar: Numeros de 0 a 255, que sean de 0 a 1
def normalize(images, labels):													#Funcion de normalización (para que el rango de valor de los píxeles sea de 0 a 1, no de 0 a 255)
    images = tf.cast(images, tf.float32)
    images /= 255																#Por eso lo dividimos entre 255
    return images, labels

train_dataset = train_dataset.map(normalize)									#Llamamos a la funcion de normalización para cada dato de ambos sets de datos
test_dataset = test_dataset.map(normalize)

#Estructura de la red
model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28,1)),								#Capa de entrada con 784 neuronas, especificando que llegará en una forma "cuadrada" de 28x28
	tf.keras.layers.Dense(64, activation=tf.nn.relu),							#Definimos dos capas ocultas densas, de 64 neuronas cada una, con la función de activación relu
	tf.keras.layers.Dense(64, activation=tf.nn.relu),
	tf.keras.layers.Dense(10, activation=tf.nn.softmax) 						#Función softmax en la última capa para clasificacion
])

#Indicar las funciones a utilizar
model.compile(
	optimizer='adam',															#Adam es un algoritmo de optimización
	loss='sparse_categorical_crossentropy',										#Funcion de coste (o función de pérdida) de entropia cruzada, usada para cuando queremos clasificar dos o más clases de etiquetas, normalmente vienen dadas como números
	metrics=['accuracy']														#Queremos obtener el dato de precisión de nuestra red, para después imprimirlo por consola
)

#Aprendizaje por lotes de 32 cada lote
BATCHSIZE = 32																				#Especificamos un tamaño de lote de 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)			#Datos de entrenamiento se entrenan de manera aleatoria
test_dataset = test_dataset.batch(BATCHSIZE)

#Realizar el aprendizaje
model.fit(
	train_dataset, epochs=5,																#Se realiza el entrenamiento. Se dan 5 vueltas a todos los datos del set
	steps_per_epoch=math.ceil(num_train_examples/BATCHSIZE) 								#Este atributo "pasos por epoca" no sera necesario en un futuro probablemente, porque vendra en versiones posteriores, si no existe ya. No es muy relevante
)

#Evaluar nuestro modelo ya entrenado, contra el dataset de pruebas
test_loss, test_accuracy = model.evaluate(
	test_dataset, steps=math.ceil(num_test_examples/32)
)

print("Resultado en las pruebas: ", test_accuracy)

#Fin del entrenamiento y de las pruebas para el modelo de la red neuronal.
#Lo que viene a continuación es para ver algunos resultados de manera gráfica. Donde podremos ver 15 ejemplos del test de evaluación donde aparece el numero en la imagen y debajo el número que predijo la red. Aparece en azul si el resultado fue correcto.

for test_images, test_labels in test_dataset.take(1):
	test_images = test_images.numpy()						#Transformamos las imagenes y las etiquetas en un array numpy de imagenes y etiquetas
	test_labels = test_labels.numpy()
	predictions = model.predict(test_images)				#model.predict() siempre espera como primer parametro un array numpy. Se obtiene un array con las predicciones

def plot_image(i, predictions_array, true_labels, images):
	predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
	plt.grid(False)					#No queremos mostrar las lineas de la cuadrícula
	plt.xticks([])					#Esta linea y la siguiente sirven para que no aparezcan unidades en los ejes de las imagenes de los numeros
	plt.yticks([])

	plt.imshow(img[...,0], cmap=plt.cm.binary)		#Mostramos la imagen con el mapa de color binario. Si cambiamos el mapa de color, cambiara el color de la imagen mostrada. NOTA: La sintaxis img[..., 0] lo que hace es omitir las primeras dimensiones del array

	predicted_label = np.argmax(predictions_array)	#Extraemos el indice del array que contiene el valor mayor
	if predicted_label == true_label:				#Si el indice del array de mayor valor coincide con la etiqueta del numero, significa que la prediccion es correcta. Si es asi, se pinta en azul, si no en rojo.
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("Prediccion: {}".format(class_names[predicted_label]), color=color)	#Imprime con el color azul o rojo "Prediccion: " y el numero que ha predicho

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#888888")				#Configura el grafico a mostrar. Rango para el numero de opciones posibles (10 numeros, 10 opciones), predictions_array con el array de probabilidades asignado a cada numero y el color inicialmente gris
	plt.ylim([0,1])																	#Se establecen limites para el eje y
	predicted_label = np.argmax(predictions_array)									#Como antes, tomamos el indice que contenga el valor mayor

	thisplot[predicted_label].set_color('red')										#Ponemos color rojo a la prediccion de nuestro modelo
	thisplot[true_label].set_color('blue')											#Ponemos color azul a la opcion correcta. (En caso de que la opcion correcta coincida con la prediccion, el color rojo se tapará por el color azul)

numrows=5								#Numero de filas que tendra nuestra figura
numcols=3								#Numero de columnas que tendra nuestra figura
numimages = numrows*numcols				#Numero total de imagenes junto con los resultados del modelo que se van a mostrar

plt.figure(figsize=(2*2*numcols, 2*numrows))		#Tamaño (en pulgadas o inches) de la figura. (ancho, altura)
for i in range(numimages):							#Bucle for que aumenta i de uno en uno
	plt.subplot(numrows, 2*numcols, 2*i+1)			#Divide la figura en numrows filas y 2*numcols columnas (la 1a columna para la imagen del numero original y la 2a columna para la grafica con la probabilidad con la que el modelo ha identificado el numero). La tercera entrada indica el numero del plot actual (el "cuadradito" en el que estamos)
	plot_image(i, predictions, test_labels, test_images)	#Se añade la imagen del numero en el "cuadradito en el que estamos"
	plt.subplot(numrows, 2*numcols, 2*i+2)			#Divide la figura en numrows filas y 2*numcols columnas (la 1a columna para la imagen del numero original y la 2a columna para la grafica con la probabilidad con la que el modelo ha identificado el numero). La tercera entrada indica el numero del plot actual (el "cuadradito" en el que estamos)
	plot_value_array(i, predictions, test_labels)			#Se añade la gráfica con la probabilidad con la que el modelo ha predicho el resultado

plt.show()	#Muestra la imagen

#Código extraído de:
#Crea tu propia red neuronal que puede leer. Ringa Tech. (https://www.youtube.com/watch?v=aFZEvQDTSyA)
#https://github.com/ringa-tech/youtube-tensorflow-mnist/

#Collaborative environment for artificial intelligence and data science (includes datasets):
#https://www.openml.org/search?type=data&sort=runs&status=active

#Some interesting links:
#https://blog.tensorflow.org/2020/01/building-ai-empowered-music-library-tensorflow.html
#https://www.tensorflow.org/datasets/catalog/fuss