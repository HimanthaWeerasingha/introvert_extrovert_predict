# Person_Introvert_Extrovert_Prediction

# Project Description

In this project build a machine learning model to predict whether a person is an introvert or extrovert based on the provided characteristics. Then deployed the model on AWS cloud platform. As a final outcome can use a public API endpoint that accepts POST requests with the same data format and returns whether the person is classified as an introvert or extrovert.

# Dataset

The dataset is balanced dataset. 

File: personality_dataset.csv

**Pre-requirtese**

Run below commands to install libraires and environment 

    ## Create python virtual environment

    python3 -m venv venv (for linux OS)

    # Activate virtual environment

    source venv/bin/activate

    # Install requirements

    pip install -r requirements.txt

# Task 1: Data Preparation

Code: data_preparation.ipynb

Load the data into pandas dataframe

Exam the dataset

Encode categorical columns in to numerical format for further analysis as below.

    "Stage_fear" column -> 'Yes': 1, 'No': 0

    "Drained_after_socializing" -> 'Yes': 1, 'No': 0

    "Personality" -> 'Introvert': 0, 'Extrovert': 1

Remove rows with missing or invalid values, assuming any missing and unwanted formated data are considered erroneous.

Outlier detection was performed using box plot visualization. Since no significant outliers were found in the dataset, no data points were removed.

Finally, saved the output into csv file

# Task 2: Model Training

Code: Model_training.ipynb

Model training was done using supervised machine learning.

Did a comparison between LogisticRegression and RandomForestClassifier using k fold method

Get the best trained model from the sleected method.

# Task 3: Test the model

Code: model_test.ipynb

Test the model mannually.

# Task 4: Deploy the model on AWS EC2 instance

Code: flask_app.py

Create AWS EC2 instance. 

To environment setup also can use requirements.txt

EC2 inbount traffic rule:

    protocol: TCP

    Port: 5000

Upload the flask_app.py and trained best model

Then run the flask_app.py

Using the public API URL that accepts POST requests can use the model (ex: post_request.jpg)

# Task 5: Example curl request

Used Flask API to receive JSON data from incoming POST request. The API accepts POST requests with the required data format and returns the predicted personality type.

curl -X POST http://<server-ip>:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"Time_spent_Alone": 4, "Stage_fear": "No", "Social_event_attendance": 6, "Going_outside": 5, "Drained_after_socializing": "Yes", "Friends_circle_size": 10, "Post_frequency": 2}'
