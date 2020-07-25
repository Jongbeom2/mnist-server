from flask import Flask, request
from flask_cors import CORS
app = Flask (__name__)
CORS(app)
import numpy as np
from PIL import Image
import glob
import pickle as pickle
from common.functions import sigmoid, softmax
##
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    
    return y

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/number', methods=['POST'])
def number():
    # save file
    f = request.files['file']
    fileName = 'tempImgFile.'+f.filename.rsplit('.', 1)[1]
    f.save(fileName)
    # predict
    network = init_network()
    global result
    for image_path in glob.glob(fileName):
        img = Image.open(image_path).convert("L")
        img = np.resize(img, (1,784))
        img = 255-(img)
        y = predict(network, img)
        result = np.argmax(y)
    return str(result)

if __name__ == "__main__":
    app.run()


