from flask import Flask, request, render_template, jsonify
import pickle
import cv2
import os.path
import skimage

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/upload",)
def upload():
    return render_template('upload.html')

@app.route("/predict", methods=['POST'])
def predict():
    f=request.files['file']
    basepath = os.path.dirname(__file__)
    filepath= os.path.join(basepath, "uploads", f.filename)
    f.save(filepath)
    print('Loading model')
    model = pickle.loads(open('model.pkl', "rb").read())
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200,200))
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = skimage.feature.hog(image, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=2, block_norm="L1")

    preds=model.predict([features])
    ls=['healthy', 'parkinson']
    result = ls[preds[0]]
    print(result)
    return jsonify({'result': result})

if __name__ == "__main__":
    app.run()










