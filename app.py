import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import keras
import numpy as np
from cv2 import cv2 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import argparse
import os





IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 250, 300, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

model = keras.models.load_model('/home/bruno/Desktop/ml_projects/santinho_predict/model-01.h5')

UPLOAD_FOLDER = '/home/bruno/Desktop/ml_projects/santinho_predict/static/img/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

labels = ['Foto Normal','Santinho Digital']

def gera_classe(img):
    img = cv2.resize(img,(150,150))
    img = np.reshape(img,[1,150,150,3])
    pred = model.predict(img)
    pred = labels[int(pred[0][0])]

    return pred

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    filename=[]
    pred =[]
    #file = []
    #file    =[]
    

    if request.method == 'POST':
        
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread(UPLOAD_FOLDER+filename)
            #gera_classe(filename)
        #file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pred = gera_classe(img)
            
    return render_template("index.html",image_folder=filename,classe=pred)


if __name__ == "__main__":  
    app.run(debug=True)   

