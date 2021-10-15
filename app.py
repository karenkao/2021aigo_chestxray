# app.py
# author:Ren
# Date: 20211001
import json
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from inference import chest_inference
app = Flask(__name__)



@app.route('/predict', methods = ['POST'])
def predict():
    json_data = request.get_json() #Get the POSTed json
    dict_data = json.loads(json_data)
    img_str = dict_data['img_str']
    result = chest_inference(img_str, tube_model_pred, pneumo_model_pred)
    return jsonify(result)

if __name__ == "__main__":
    # parameters
    tube_weight_path = "./weights/tube/06_chest_npz_aug10_zero2one_512x512_valWOaug_clahe_centercrop/checkpoint-20-0.985-0.040.h5"
    pneumo_weight_path = "./weights/pneumo/01_pneumo_npz_aug10_normalize1024_512x512_valWOaug_densenet/checkpoint-24-0.951-0.247.h5"
    # load tube model
    tube_model_pred = load_model(tube_weight_path)
    # load pneumo model
    pneumo_model_pred = load_model(pneumo_weight_path)
    app.run(host = "0.0.0.0", port = 8080)
