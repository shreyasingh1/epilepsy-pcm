import numpy as np
import pandas as pd
import json
from epilepsypcm.utils.outcome_params import seizure_onset_zone, engel_score

# INPUT
# patient = string format, patient number
# paths = path to CCEP response files, in os format
# OUTPUT
# df = dataframe for one patient with
#       X features with columns: chNames, significant, n1, n2, p2 z scores,
#       n1, n2, p2 latencies, and flipped
#       and associated y outcome labels
def make_df(patient, paths):
    #extracting info from each response file
    n = 0
    for i in range(len(paths)):
        chNames = []
        # load info into python dictionary
        data = json.load(open(paths[i]))

        # Get list of channel names
        for key in data["time"]: chNames.append(key)

        # loop over each channel, and extract average time series and information about the peaks

        if n < 1:
            avgResp = np.empty((len(paths), len(chNames), len(data['time'][chNames[0]])))
            significant = np.empty((len(paths), len(chNames)))
            n1Zscore = np.empty((len(paths), len(chNames)))
            n2Zscore = np.empty((len(paths), len(chNames)))
            p2Zscore = np.empty((len(paths), len(chNames)))
            n1Latency = np.empty((len(paths), len(chNames)))
            n2Latency = np.empty((len(paths), len(chNames)))
            p2Latency = np.empty((len(paths), len(chNames)))
            flipped = np.empty((len(paths), len(chNames)))
            n += 1
            samplingRate = np.empty((len(paths)))
            window = np.empty((len(paths), 2))

        for j in range(len(chNames)):
            avgResp[i][j] = data['time'][chNames[j]]
            significant[i][j] = data['significant'][chNames[j]]
            n1Zscore[i][j] = data['zscores'][chNames[j]]['n1'][1]
            n2Zscore[i][j] = data['zscores'][chNames[j]]['n2'][1]
            p2Zscore[i][j] = data['zscores'][chNames[j]]['p2'][1]
            n1Latency[i][j] = data['zscores'][chNames[j]]['n1'][0] + data['window'][0] * data["samplingRate"] / 1000
            n2Latency[i][j] = data['zscores'][chNames[j]]['n2'][0] + data['window'][0] * data["samplingRate"] / 1000
            p2Latency[i][j] = data['zscores'][chNames[j]]['p2'][0] + data['window'][0] * data["samplingRate"] / 1000
            flipped[i][j] = data['zscores'][chNames[j]]['flipped']

        samplingRate[i] = data["samplingRate"]
        window[i] = data['window']

    # creating dataframe

    df = pd.DataFrame()
    df["chNames"] = chNames * len(paths)
    df["significant"] = significant.flatten()
    df["n1Zscore"] = n1Zscore.flatten()
    df["n2Zscore"] = n2Zscore.flatten()
    df["p2Zscore"] = p2Zscore.flatten()
    df["n1Latency"] = n1Latency.flatten()
    df["n2Latency"] = n2Latency.flatten()
    df["p2Latency"] = p2Latency.flatten()
    df["flipped"] = flipped.flatten()

    # Dropped rows for stimulating channels since they only
    # contain stimulating waveforms / artifacts / saturated signals
    # Also zero out rows with latency values of -999.0

    # drop rows in the dataframe with latency values of -999.0
    df = df.drop(df.loc[df["n1Latency"] == -999.0].index)
    df = df.drop(df.loc[df["n1Latency"] == -499.0].index)

    # drop rows in the dataframe for stimulated channels
    for i in range(len(paths)):
        # get stimulated channels
        fileInfo = paths[i].split("_")
        stimCh1 = fileInfo[1];
        stimCh2 = fileInfo[2];
        stimCh = stimCh1 + "_" + stimCh2
        df = df.drop(df.loc[df["chNames"] == stimCh].index)

    # adding dataframe outcome values (1 if in SOZ, 0 if not)
    df["outcome"] = np.zeros(df.shape[0])

    if engel_score[patient] == "1":
        if seizure_onset_zone[patient] != "None":
            for node in seizure_onset_zone[patient]:
                for channel in df["chNames"]:
                    if node in channel:
                        df["outcome"][df["chNames"] == channel] = 1

    return df

# Function that takes in the location of all patient folders and engel
# score of interest, and returns a nested list of dataframes for each patient
# INPUT:
# base_path = string, file location to base folder that contains all patient folders
# engel_score = string, target engel score to get dataframe for (ex. "1")
#               can currently only handle "1" and "2"
# OUTPUT:
# positive_dataframes = a nested list, where [patient number (string), dataframe].

import glob
import os
from pathlib import Path

def get_df_list(base_path, engel):
    patient_files = os.listdir(base_path)

    positive_dataframes = []
    for file in patient_files:
        if (file[0] == "P") & (file != "PY16N006"):
            response_path = base_path + file + '/ResponseInfo/CCEP'
            response_files_path = glob.glob(response_path + '/*.json', recursive=True)

            # Getting individual dataframe for positive patients
            patient = file
            if file in engel_score.keys():  # if we currently have the file's engel score
                if engel_score[patient] == engel:  # if the engel score is 1
                    df = make_df(patient, response_files_path)
                    positive_dataframes.append([patient, df])

    return positive_dataframes

def concat_dfs(base_path, engel):
    patient_files = os.listdir(base_path)

    full_df = pd.DataFrame()
    for file in patient_files:
        if (file[0] == "P") & (file != "PY16N006"):
            response_path = base_path + file + '/ResponseInfo/CCEP'
            response_files_path = glob.glob(response_path + '/*.json', recursive=True)

            # Getting individual dataframe for positive patients
            patient = file
            if file in engel_score.keys():  # if we currently have the file's engel score
                if engel_score[patient] == engel:  # if the engel score is 1
                    df = make_df(patient, response_files_path)
                    full_df = pd.concat([full_df, df])

    return full_df