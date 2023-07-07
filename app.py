import os
import csv
from flask import Flask, request, render_template
import pandas as pd

import preprocess #Python file containing pre-defined functions
import zipfile
import pyzipper
import joblib

import warnings
warnings.filterwarnings("ignore")



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the 'file' field is in the request
    if 'file' not in request.files:
        return 'No file uploaded'
    
    file = request.files['file']
    
    # Check if a file is selected
    if file.filename == '':
        return 'No file selected'

    # Check if the file is a CSV file
    if not file.filename.endswith('.zip'):
        return 'Invalid file format. Please upload a ZIP file.'

    try:
        # Save the uploaded CSV file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        ativity_stage = preprocess.zip_to_csv(file_path)
        time_list, activity_pred = preprocess.predict_activity(ativity_stage)
        print(time_list)

        table_data = []  # Initialize an empty list to store table rows
        for i in range(0, len(time_list)):
            my_dict = {'Time':time_list[i], 'Activity':activity_pred[i]}
            table_data.append(my_dict)


        return render_template('index.html', table_data=table_data)#'ZIP file successfully uploaded and processed.'

    except Exception as e:
        return 'An error occurred during file upload: ' + str(e)

if __name__ == '__main__':
    app.run(debug = True)
