from flask import Flask
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# variables Flask
app = Flask(__name__)
api = Api(app)


# se carga el modelo de Logistic Regression del Notebook #3
pkl_filename = "ModeloLR.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

# Load MNIST model
pkl_mnist = 'mnist_model.pkl'
with open(pkl_mnist, 'rb') as file:
    model_mnist = pickle.load(file)

parser = reqparse.RequestParser()
parser.add_argument('petal_length')
parser.add_argument('petal_width')
parser.add_argument('sepal_length')
parser.add_argument('sepal_width')

number_parser = reqparse.RequestParser()
number_parser.add_argument('img')

size = 28, 28

def decode_img(encoded_string):
    img = Image.open(BytesIO(base64.b64decode(encoded_string)))
    return img  

def reshape_image(img):
    img.thumbnail(size, Image.ANTIALIAS) # resize
    img =  np.invert(img.convert('L')).ravel() # Convert to grayscale and set shape as (28,)

    img = img / 255

    return img
    

class Predict(Resource):

    @staticmethod
    def post():
        # request para el modelo
        args = parser.parse_args() 
        datos = np.fromiter(args.values(), dtype=float) 

        # prediccion
        out = {'Prediccion': int(model.predict([datos])[0])}

        return out, 200

    # TODO: Define el def get()
    # ejercicio semanal

    def get(self):
        args = parser.parse_args() 
        datos = np.fromiter(args.values(), dtype=float) 

        # prediccion
        out = {'Prediccion': int(model.predict([datos])[0])}

        return out, 200

class PredictNumber(Resource):
    @staticmethod
    def post():
        args = number_parser.parse_args()
        img = decode_img(args.img)
        img = reshape_image(img)

        result = model_mnist.predict([img])[0]
        print(result)
        # result = 44

        return { 'result': result },200

    def get(self):
        return { 'ola': 'hola get'}, 200

api.add_resource(Predict, '/predict')
api.add_resource(PredictNumber, '/predict-number')

if __name__ == '__main__':
    app.run(debug=True, port='1080')