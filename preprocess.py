import zipfile
import pyzipper
import pandas as pd
import joblib

import warnings
warnings.filterwarnings("ignore")

def zip_to_csv(zip_dir, password = 'dkMreHXt'):
    #Extract
    with pyzipper.AESZipFile(zip_dir) as zip_ref:
        # Set the password to decrypt the zip file
        zip_ref.setpassword(password.encode())
        # Extract all files and folders
        zip_ref.extractall()
    csv_paths = [] #To save path to csv files
    # Open the zip file
    with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
        # Get a list of file names within the zip file
        file_names = zip_ref.namelist()
        
        # Iterate over the file names
        for file_name in file_names:
            # Check if the file is a CSV file
            if file_name.endswith('.csv'):
                csv_paths.append(file_name)
                #print(file_name)

    #Read csv files
    #We are interested in 2 csv files, 'ACTIVITY_STAGE' csv and 'HEARTRATE_AUTO' csv
    for csv in csv_paths:
        if csv.split('/')[0] == 'ACTIVITY_STAGE': #if the first word before '/' is 'ACTIVITY_STAGE'
            ativity_stage = pd.read_csv(csv)
        elif csv.split('/')[0] == 'HEARTRATE_AUTO':
            heart_rate = pd.read_csv(csv)

    #Next, we add average heart rate colunm for activities in activity_stage df
    heart_rate['time'] = pd.to_datetime(heart_rate['time'])
    heart_rate_meanlist = []
    for i in range(0, len(ativity_stage)):
        date = ativity_stage.loc[i]['date']
        start = ativity_stage.loc[i]['start']
        stop = ativity_stage.loc[i]['stop']
        date_heartrate = heart_rate[heart_rate['date'] == date]
        # Define the start and end time for the desired range
        start_time = pd.to_datetime(start).time()
        end_time = pd.to_datetime(stop).time()
        # Create a boolean mask for values within the specified time range
        time_mask = (date_heartrate['time'].dt.time >= start_time) & (date_heartrate['time'].dt.time <= end_time)
        # Apply the mask to filter the DataFrame
        filtered_df = date_heartrate[time_mask]
        hrate_mean = filtered_df['heartRate'].mean()
        heart_rate_meanlist.append(hrate_mean)
    ativity_stage['mean_heartrate'] = heart_rate_meanlist
    return ativity_stage

#Testing the function
#zip_dir = "3089914011_1685273976448.zip"
#ativity_stage = zip_to_csv(zip_dir)
#print('Ativity Stage\n', ativity_stage.head()) #Ativity Stage

def predict_activity(ativity_stage):
    x = ativity_stage.drop(['date', 'start', 'stop'], axis=1)
    ## Load model and Predict
    loaded_model = joblib.load('rf_clf.joblib')
    preds = loaded_model.predict(x)
    #Create a column for predictions
    ativity_stage['predictions'] = preds 
    ativity_stage['predictions'] = ativity_stage['predictions'].replace({0: 'Cycling', 1: 'Running', 2: 'walking', 3: 'Relaxing',
                                                                         4: 'Reading', 5: 'Preparing a meal'})
    test = ativity_stage[ativity_stage['date'] == '2023-04-12']

    #Determine Activity being done at a specific hour
    # Convert 'start_time' and 'stop_time' columns to datetime objects
    test['start_time'] = pd.to_datetime(test['start'])
    test['stop_time'] = pd.to_datetime(test['stop'])
    # Extract the hour component from 'start_time' and 'stop_time' columns
    test['start_hour'] = test['start_time'].dt.hour
    test['stop_hour'] = test['stop_time'].dt.hour

    time_list = list(test['start_hour'].unique()) #We use 'unique' to avoid duplicates
    # Filter the DataFrame for rows where 'start_hour' or 'stop_hour' is equal to start time
    activity_pred = []
    for i in time_list:
        filtered_df = test[(test['start_hour'] == i) | (test['stop_hour'] == i)]
        # Print the filtered DataFrame
        my_pred = filtered_df['predictions'].mode()
        activity_pred.append(my_pred.values[0])
    return time_list, activity_pred #Returns a list of classified events alongside a list of time when the activity took place

## Test the function
#time_list, activity_pred = predict_activity(ativity_stage)
#print('Time list\n', time_list) #Time
#print('Activities\n', activity_pred) #Time
