import pickle
from flask import Flask
from flask import request
from flask import jsonify

import logic_of_web_service

model_file = 'modellog5.bin'

with open(model_file,'rb') as f_in:
    dv,model= pickle.load(f_in)

app = Flask('predict risk of stroke')
@app.route('/predict', methods=['POST'])
def prediction():
    patient= request.get_json()
    y_pred, stroke_risk = logic_of_web_service.logic_web(patient,dv,model)
    result = {
        'stroke_prediction': float(y_pred),
        'risk_likelihood':'High stroke risk likelihood'if 0.5<=y_pred else 'Medium stroke risk likelihood' if 0.5>y_pred>=0.4 else 'Low stroke risk likelihood'
    }
    return jsonify(result)


if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port= 7777)