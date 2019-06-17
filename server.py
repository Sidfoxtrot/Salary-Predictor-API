import numpy as np
from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/predict',methods=['POST'])
def predict():
# Getting the data from the POST request.
    data = request.get_json(force=True)
#prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['exp'])]])
# Taking the first value of prediction
    out = prediction[0]
    return jsonify(out)
if __name__ == '__main__':
    app.run(port=5000, debug=True)
