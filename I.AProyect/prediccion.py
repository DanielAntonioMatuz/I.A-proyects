import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 32, 32
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: AVION")
  elif answer == 1:
    print("pred: AUTOMOVIL")
  elif answer == 2:
    print("pred: AVE")
  elif answer == 3:
    print("pred: GATO")
  elif answer == 4:
    print("pred: VENADO")
  elif answer == 5:
    print("pred: PERRO")
  elif answer == 6:
    print("pred: RANA")
  elif answer == 7:
    print("pred: CABALLO")
  elif answer == 8:
    print("pred: BARCO")
  elif answer == 9:
    print("pred: TRAILER")

  return answer

predict("./data/1998.jpg")
