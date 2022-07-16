from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from skimage import io
from keras.models import load_model
import cv2
from PIL import Image #use PIL
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        print("File Received")
        filename = secure_filename(file.filename)
        file.save(app.config("static/"+ filename)) #Heroku no need static
        file = open(app.config("static/"+ filename,"r")) #Heroku no need static
        model = load_model("Pneumonia")
        img = Image.open(filename) #rose = 3, sunflower = 4, tulip 5
        img = img.resize((100,100))
        img = np.asarray(img, dtype="float32") #need to transfer to np to reshape
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) #rgb to reshape to 1,100,100,3
        img.shape
        pred=model.predict(img)
        return(render_template("index1.html", result=str(pred)))
    else:
        return(render_template("index1.html", result="2"))
    
if __name__ == "__main__":
    app.run()
