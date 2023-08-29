# Artificial-Intelligence
Artificial Intelligence.

- FaceRecognitionV1: Contains a basic program in Python that recognizes pictures of human faces using the OpenCV library and the face_recognition module.

      Its purpose is to make a first approach to learning the face recognition basics at a high-level programming as well as getting familiar with Python.

      In order to use the program two folders need to be added to the FaceRecognitionV1 folder:
  
      1. Folder named "Known_Identities": Containing images (.jpg format) of ONE SINGLE PERSON (as though it is an official ID picture). If possible, name the image in such way that it would be easy for you to find it, for example with the ID number (e.g: 00000023T).
  
      2. Folder named "Unknown_Identities": Containing images(.jpg format) of UNKNOWN PEOPLE (can be none or many unidentified people in the image). The program will compare all the known identities faces with all the unknown people in each picture. The output of the program will be each picture with the unknown identities framed in green if recognized.

      NOTE: It may not work for windows users. If so, try changing the path of the folders within the program code.

- OpenCVLearning: Contains a basic program in Python that recognizes human faces with the computer camera in real time using the OpenCV library and the face_recognition module.

      Its purpose is to give an example on how to read and work with video frames and get the camera working using OpenCV.

      In order to use the program one folder needs to be added to the OpenCVLearning folder:

      1. Folder named "Known_Identities": Containing images(.jpg format) of ONE SINGLE PERSON (as though it is an official ID picture). Those would be the faces that the program will try to recognize in real time.

      NOTE: It may not work for windows users. If so, try changing the path of the folders within the program code.

- Ejemplo_Red_Neuronal_1: Python program that uses Tensorflow and the MNIST dataset to train a model than is capable of predicting a handwritten number.

      Its purpose is to learn the basics of using Tensorflow, loading a dataset, training a model and testing it. As well as getting familiar with some other libraries such as numpy or matplotlib.