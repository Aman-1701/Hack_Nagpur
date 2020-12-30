import joblib
from flask import Flask, request, jsonify
from Predict.json_heart import predict_result


app = Flask('mod-heart')

@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    model = joblib.load(r'./model_files/heart.pkl')
    result = predict_result(data, model)

    return jsonify(result)


#For local deployment

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    
