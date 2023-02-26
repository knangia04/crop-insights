from flask import Flask, Response, render_template, url_for, request
import cv2
import pickle
import numpy as np
import requests
import pandas as pd

app = Flask(__name__)

camera = cv2.VideoCapture(0)


import requests

url = "https://plants2.p.rapidapi.com/api/plants"

headers = {
	"Authorization": "GKZOHNZj0xP65kk0BAE2Tl9LGagm0pfD3DFNxAEEZcMQBhRZVDco8vbNJdnwwCo0",
	"X-RapidAPI-Key": "f792eee498mshce2f349f903dddap1dbd70jsnd9e53c2fe728",
	"X-RapidAPI-Host": "plants2.p.rapidapi.com"
}


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

y = "crop_yield_prediction.pickle"
yield_model = pickle.load(open(y, "rb"))

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
    if request.method == 'POST':
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
    crop = targets[crop_encoded[0]] # post processing

    querystring = {"CN":crop}

    response = requests.request("GET", url, headers=headers, params=querystring)

    print(response.text)
    info = response.json()
    c = 0
    for i in info:
        if ['TemperatureMinimum', 'pH_Minimum', 'MatureHeight', 'Precipitation_Minimum', 'Precipitation_Maximum','pH_Maximum', 'TemperatureMaximum'] in list(i.keys()):
            if int(i['TemperatureMinimum']) != 0 or int(i['pH_Minimum']) != 0 or int(i['MatureHeight']) != 0 or int(i['Precipitation_Minimum']) != 0 or int(i['Precipitation_Maximum']) != 0 or int( i['pH_Maximum']) != 0 or int(i['TemperatureMaximum']) != 0:
                info = info[c]
                break
            c+=1
    try:
        info = info[c] 
    except:
        info = []

    # convert the prediction to the correct label after it was encoded 
    print(crop)
    return render_template('insights.html', crop=crop.upper(), info=info)

@app.route('/crop_recommendation', methods=['POST', 'GET'])
def insights():
    return render_template('insights.html')



@app.route('/analyze_yield', methods=['POST'])
def analyze_yield():
    if request.method == 'POST':
         # get the data from the form 
        rain = request.form['rain']
        temp = request.form['temp']
        pesticides = request.form['pesticides'] 
        print(rain, temp, pesticides)

    e = pd.DataFrame([[rain, pesticides, temp]],columns=['average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp'])
    pred = yield_model.predict(e)

    # convert the prediction to the correct label after it was encoded 
    print(pred)
    return render_template('yield_insights.html', info=pred, rain=rain, temp=temp, pesticides=pesticides)
    

@app.route('/yield_prediction', methods=['POST', 'GET'])
def yield_insights():
    return render_template('yield_insights.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
