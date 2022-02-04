import json
import glob
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from epilepsypcm.loading.imports import respInfoToAdjacencyMatrix, zToAdjacencyMatrix, getPeakLatency


#still need to write code that loops through all patient folders
##but for now, just starting with the data in one folder

base_path = '/Users/zhonglelin/Files/02研究生学习资料/研一/04Precision Care Medicine/Epilepsy/data/PY16N006' #modify for your file location
# files= os.listdir(base_path)
# print(files)
# for i in range()
response_path = base_path + '/ResponseInfo/CCEP'
z_path = base_path + '/z-scores'

response_files = os.listdir(response_path)
response_files_path = glob.glob(response_path + '/*.json', recursive=True)

z_files = os.listdir(z_path)
z_files_path = glob.glob(z_path + '/*.json', recursive=True)

# EXAMPLE opening response files dictionary and indexing
k = open(response_files_path[3])
data = json.load(k)
keys = []
for key in data["zscores"]: keys.append(key)
print(keys)

A, nodeLabels, allStimInds, chNames = respInfoToAdjacencyMatrix(response_files_path, "n1", 6)

import seaborn as sns
import matplotlib.pyplot as plt

A_positive = A
A_positive[A_positive<0] = 0

plt.figure(figsize = (15,12))
sns.heatmap(A_positive, cmap = "mako", xticklabels = chNames, yticklabels = chNames)

A, nodeLabels, allStimInds, chNames = zToAdjacencyMatrix(z_files_path)

A_positive = A
A_positive[A_positive<0] = 0

plt.figure(figsize = (15,12))
sns.heatmap(A_positive, cmap = "mako", xticklabels = chNames, yticklabels = chNames)

chNames, L = getPeakLatency(response_files_path, "n1")

# extracting info from each response info file
paths = response_files_path

n = 0
for i in range(len(paths)):
    chNames = []
    # load info into python dictionary
    fileInfo = paths[i].split("_")
    stimCh1 = fileInfo[1];
    stimCh2 = fileInfo[2];
    jsonData = json.load(open(paths[i]))
    
    #Get list of channel names
    for key in jsonData["time"]: chNames.append(key)
        
    # loop over each channel, and extract average time series and information about the peaks

    if n < 1:
        avgResp = np.empty((len(paths),len(chNames),len(data['time'][chNames[0]])))
        significant = np.empty((len(paths),len(chNames)))
        n1Zscore = np.empty((len(paths),len(chNames)))
        n2Zscore = np.empty((len(paths),len(chNames)))
        p2Zscore = np.empty((len(paths),len(chNames)))
        n1Latency = np.empty((len(paths),len(chNames)))
        n2Latency = np.empty((len(paths),len(chNames)))
        p2Latency = np.empty((len(paths),len(chNames)))
        flipped = np.empty((len(paths),len(chNames)))
        n += 1
        samplingRate = np.empty((len(paths)))
        window = np.empty((len(paths),2))
        
        
    for j in range(len(chNames)):
        avgResp[i][j] = data['time'][chNames[j]]
        significant[i][j] = data['significant'][chNames[j]]
        n1Zscore[i][j] = data['zscores'][chNames[j]]['n1'][1]
        n2Zscore[i][j] = data['zscores'][chNames[j]]['n2'][1]
        p2Zscore[i][j] = data['zscores'][chNames[j]]['p2'][1]
        n1Latency[i][j] = data['zscores'][chNames[j]]['n1'][0]+data['window'][0]*data["samplingRate"]/1000
        n2Latency[i][j] = data['zscores'][chNames[j]]['n2'][0]+data['window'][0]*data["samplingRate"]/1000
        p2Latency[i][j] = data['zscores'][chNames[j]]['p2'][0]+data['window'][0]*data["samplingRate"]/1000
        flipped[i][j] = data['zscores'][chNames[j]]['flipped'] 
        
    samplingRate[i] = data["samplingRate"]
    window[i] = data['window']
    
    # set the traces of stimulating channels to zero since they only
    # contain stimulating waveforms / artifacts / saturated signals
    #avgResp{k}(find( any(strcmp([split(chNames{k},'_')'],stimCh1{k})) | any(strcmp([split(chNames{k},'_')'],stimCh2{k}))))= 0;    
