from flask import Flask, Response, render_template, url_for, request
import cv2
import pickle
import numpy as np
import pandas as pd
# import tensorflow.compact.v2 as tf
# import keras
# from sklearn.externals import joblib

app = Flask(__name__)

camera = cv2.VideoCapture(0)

# model = joblib.load_model('cnn_model.pkl')

# model = pickle.load(open('cnn_model.pkl', 'rb'))
# print(model.predict())
# print('Model loaded. ')


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
    
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            # print(model.predict(x=[convert_image_to_array(buffer)]))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

df=pd.read_csv('Crop_recommendation.csv')
c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes
model = "my_model.pickle"
grad = pickle.load(open(model, "rb"))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/resources")
def resources():
    return render_template('resources.html')

@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/disease_detection', methods=['POST', 'GET'])
def disease_detection():
    return render_template('disease_detection.html')

# route for insights page 
@app.route('/analyze', methods=['POST'])
def analyze():
    if reqeust.method == 'POST':
        # get the data from the form 
        N = request.form['N']
        P = request.form['P']
        K = request.form['K']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']

    entry = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],columns=['N','P','K','temperature','humidity','ph','rainfall'])
    crop_encoded = grad.predict(entry)
    crop = targets[crop_encoded[0]]

    # convert the prediction to the correct label after it was encoded 
    print(targets[crop[0]])
    return url_for('crop_recommendation')

@app.route('/crop_recommendation', methods=['POST', 'GET'])
def crop_recommendation():
    return render_template('insights.html')

@app.route('/crop_recommendation/<crop>', methods=['POST', 'GET'])
def crop_recommendation(crop):
    crop = 'rice'
    return render_template('insights.html', crop=crop)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
