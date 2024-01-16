from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

# Load the submission file to get predictions
submission_data = pd.read_csv("G:\Savi\capstone\ML_model_deployment\submission.csv")
print(submission_data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_id = int(request.form.get('id', 0))

    # Check if the input_id is within the specified range
    if 1315 <= input_id <= 2631:
        # Find the corresponding prediction for the given id
        prediction_row = submission_data[submission_data['id'] == input_id]

        # Debug information
        print("Received input id:", input_id)
        print("Prediction row:", prediction_row)

        # Check if the prediction exists
        if not prediction_row.empty:
            prediction = prediction_row['rent'].values[0]
            print("Prediction:", prediction)
            return render_template('index.html', prediction=prediction)
        else:
            print("Id not found")
            return render_template('index.html', prediction="Id not found")
    else:
        print("Invalid input id:", input_id)
        return render_template('index.html', prediction="Invalid input id")

if __name__ == '__main__':
    app.run(debug=True)
