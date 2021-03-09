import numpy as np
import cv2

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image

# cargar clasificador pre-entrenado
cascada_caras = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

# cargar modelo pre-entrenado
modelo = tf.keras.models.load_model('reconcedor-facial.h5')

# emociones
etiquetas = ['ira','contento','disgusto','miedo','feliz','tristeza','sorpresa']

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteccion de caras
    caras = cascada_caras.detectMultiScale(gray, 1.3, 5)

    # Cara Actual
    actual = None


    archivo = "test.jpg"
    
    img = image.load_img(archivo, target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    imagenes = np.vstack([x])
    clases = modelo.predict(imagenes, batch_size=1)
    indice = clases.tolist()[0].index(1.0)
    print(etiquetas[indice])






    # usamos cv2.rectangle para poner un rectangulo a cada
    # cara registrada
    for (x,y,w,h) in caras:
        # se sobre escribe la image en cada ciclo
        gray = cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),10)
        actual = frame[y:y+h, x:x+w]

    # Deteccion con Keras ===================================
    if(actual is not None):

        


        #actual = cv2.resize(actual, (48,48), interpolation = cv2.INTER_AREA)
        #x = image.img_to_array(actual)
        #x = np.expand_dims(x, axis=0)
        #x = np.vstack([x])
        #print(x.shape)
        #cv2.imwrite('test.jpg',actual)
        #clases = modelo.predict(x, batch_size=1)
        #indice = clases.tolist()[0].index(1.0)

        # Escribir emocion en la pantalla
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(gray, etiquetas[indice], (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)

    # =======================================================

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()