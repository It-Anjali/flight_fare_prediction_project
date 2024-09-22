from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Keep only one definition for the home route
@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')

# Load the model once at the beginning
model = pickle.load(open("flight_rf.pkl", "rb"))

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        # Extract and process form data
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").month)

        Dep_hour = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").minute)

        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").minute)

        Duration = abs((Arrival_hour - Dep_hour) * 60 + (Arrival_min - Dep_min))
        Total_stops = int(request.form["stops"])

        # Airline data processing
        airline = request.form['Airline']
        airline_dict = {
            'Jet Airways': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IndiGo': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Air India': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'Multiple carriers': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'SpiceJet': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'Vistara': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'GoAir': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'Multiple carriers Premium economy': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'Jet Airways Business': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'Vistara Premium economy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'Trujet': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
        
        airline_encoded = airline_dict.get(airline, [0] * 11)

        # Source data processing
        source_dict = {
            'Delhi': [1, 0, 0, 0],
            'Kolkata': [0, 1, 0, 0],
            'Mumbai': [0, 0, 1, 0],
            'Chennai': [0, 0, 0, 1]
        }

        source_encoded = source_dict.get(request.form["Source"], [0, 0, 0, 0])

        # Destination data processing
        destination_dict = {
            'Cochin': [1, 0, 0, 0, 0],
            'Delhi': [0, 1, 0, 0, 0],
            'New_Delhi': [0, 0, 1, 0, 0],
            'Hyderabad': [0, 0, 0, 1, 0],
            'Kolkata': [0, 0, 0, 0, 1]
        }

        destination_encoded = destination_dict.get(request.form["Destination"], [0, 0, 0, 0, 0])

        # Create a DataFrame with all the processed inputs
        input_data = pd.DataFrame([[
            Duration,
            Total_stops,
            Journey_day,
            Journey_month,
            Dep_hour,
            Dep_min,
            Arrival_hour,
            Arrival_min,
            *airline_encoded,
            *source_encoded,
            *destination_encoded
        ]])

        # Predict and return the result
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)
        
        return render_template("home.html", prediction_text="Your Predicted Flight Fare is Rs. {}".format(output))
    
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
