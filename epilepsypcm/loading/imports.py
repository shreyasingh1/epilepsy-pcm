#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

#IMPORTS
import numpy as np
import json
from statistics import mean

# Function to read in a full list, and a list of terms to
# get the index of. Returns a list of indexes in the full list
# where the search terms are present
# @args full_list
# @args search_term, list of terms to index for
# @returns a list of overlap indexes, and [-1] if no overlap
def getIndxOfOverlap(full_list, search_term):
    toReturn = []
    count = full_list.count(search_term)

    # for i in range(count):
    try:
        indx = full_list.index(search_term)
        toReturn.append(indx)
    except:
        x = 1

    for i in range(count - 1):
        indx = toReturn[i]
        toReturn.append(indx + 1)

    if len(toReturn) > 0:
        return toReturn
    else:
        return [-1]

### Function to produce respInfo to adjacency matrix

# INPUTS:
# files = list of file paths
# peaks = string, either n1, n2, p2, or pk2pk (amplitude from n1 to p2)
# threshold = threshold for including the peak in the adjacency matrix

# RETURNS:
# A = adjacency matrix containing response magnitudes of peak of interest
#       specified by input parameter 'peaks'. Stimulating nodes are on the
#       rows, and response nodes are on the columns.
# nodeLabels = the names of the nodes in the adjacency matrix, as a vector
#       in the same order as they appear in 'A'
# allStimInds = vector of nodes indicies that were stimulated
# chNames = list of all channel names in file
def respInfoToAdjacencyMatrix(files, peaks, threshold):
    responseChs = [];
    peakScores = [];
    stimChs = [];

    for k in files:
        chNames = []
        # load info into python dictionary
        fileInfo = k.split("_")
        stimCh1 = fileInfo[1];
        stimCh2 = fileInfo[2];
        jsonData = json.load(open(k))
        for key in jsonData["zscores"]: chNames.append(key)

        # loop over each response channel and build lists
        for i in range(len(chNames)):
            peakValues = [];  # why is this here
            if "pk2pk" == peaks:
                peakAmplitude = jsonData["zscores"][chNames[i]]["n1"][1] + jsonData["zscores"][chNames[i]]["p2"][1]
            else:
                peakAmplitude = abs(jsonData["zscores"][chNames[i]][peaks][1])

            # peakAmplitude from float to int
            # peakAmplitude = int(peakAmplitude)

            if (peakAmplitude > threshold):
                # if ((len(peakAmplitude) == 1) & (peakAmplitude > threshold)) | len(peakValues) > 1:
                responseChs.append(chNames[i])
                peakScores.append(peakAmplitude)  # peakValues, not peakAmplitude
                stimChs.append(stimCh1 + "_" + stimCh2)

    # Build Adjacency Matrix and Graph
    allStimInds = [];
    nodeLabels = chNames;
    nodeLabels.sort()
    stimChs.sort()

    A = np.zeros((len(chNames), len(chNames)));
    # loop over each stimulation block
    for i in range(len(stimChs)):
        stimInds = getIndxOfOverlap(nodeLabels, stimChs[i])
        if stimInds != [-1]:
            allStimInds.append(stimInds);

        respInd = getIndxOfOverlap(nodeLabels, responseChs[i])

        if len(respInd) == 1 & respInd[0] == 0:  # if no responses, continue
            continue

        # get indicies of all responses that were stimulated in this block
        ind = getIndxOfOverlap(stimChs, stimChs[i])

        # loop over each response and store the zscore in the right spot in the adjacency matrix
        for r in range(len(respInd)):
            r1 = getIndxOfOverlap([responseChs[i] for i in ind], nodeLabels[respInd[r]])[0]

            if len(stimInds) < 1:
                A[stimInds, respInd[r]] = peakScores[ind[r1[len(r1) - 1]]];
                continue

            if (A[stimInds, respInd[r]] > 0) & (peakScores[ind[r1]] > 0):
                A[stimInds, respInd[r]] = mean([A[stimInds, respInd[r]][0], peakScores[ind[r1]]])
            elif (A[stimInds, respInd[r]] == 0) & (peakScores[ind[r1]] > 0):
                A[stimInds, respInd[r]] = peakScores[ind[r1]]

    allStimIndsUQ = []
    for i in allStimInds:
        if len(i) > 0:
            allStimIndsUQ.append(i[0])
    allStimIndsUQ = (set(allStimIndsUQ));  # only get the unique stimulated pairs

    A = A - np.diag(A)  # make sure there is no activity on the diagonal

    return A, nodeLabels, allStimIndsUQ, chNames


### Function for z-score files to adjacency matrix

# INPUTS:
# files = list of file paths

# RETURNS:
# A = adjacency matrix containing N1 z-scores. Stimulating nodes are on the
#       rows, and response nodes are on the columns.
# nodeLabels = the names of the nodes in the adjacency matrix, as a vector
#       in the same order as they appear in 'A'
# allStimInds = vector of nodes indicies that were stimulated
# chNames = all channels in the input files
def zToAdjacencyMatrix(files):
    responseChs = [];
    stimChs = [];
    zscores = [];

    for k in files:
        chNames = []
        # load info into python dictionary
        fileInfo = k.split("_")
        stimCh1 = fileInfo[1];
        stimCh2 = fileInfo[2];
        # jsonData=jsondecode(fileread(strcat(files(k).folder,"\\",files(k).name)));
        jsonData = json.load(open(k))
        # chNames = fieldnames(jsonData.zscores);
        for key in jsonData.keys(): chNames.append(key)

        # loop over each response channel and build lists
        for i in chNames:
            Z = jsonData[i]
            if (len(str(Z)) == 1 & int(Z) > 0.0) | len(str(Z)) > 1:
                responseChs.append(i)
                zscores.append(Z)
                stimChs.append(str(stimCh1) + "_" + str(stimCh2))

    # Build Adjacency Matrix and Graph
    allStimInds = [];
    nodeLabels = chNames;
    nodeLabels.sort()
    stimChs.sort()

    A = np.zeros((len(chNames), len(chNames)));
    # loop over each stimulation block
    for i in range(len(stimChs)):
        stimInds = getIndxOfOverlap(nodeLabels, stimChs[i])
        if stimInds != [-1]:
            allStimInds.append(stimInds);

        respInd = getIndxOfOverlap(nodeLabels, responseChs[i])

        if len(respInd) == 1 & respInd[0] == 0:  # if no responses, continue
            continue

        # get indicies of all responses that were stimulated in this block
        ind = getIndxOfOverlap(stimChs, stimChs[i])

        # loop over each response and store the zscore in the right spot in the adjacency matrix
        for r in range(len(respInd)):
            r1 = getIndxOfOverlap([responseChs[i] for i in ind], nodeLabels[respInd[r]])[0]

            if len(stimInds) < 1:
                A[stimInds, respInd[r]] = zscores[ind[r1[len(r1) - 1]]];
                continue

            if (A[stimInds, respInd[r]] > 0) & (zscores[ind[r1]] > 0):
                A[stimInds, respInd[r]] = mean([A[stimInds, respInd[r]][0], zscores[ind[r1]]])
            elif (A[stimInds, respInd[r]] == 0) & (zscores[ind[r1]] > 0):
                A[stimInds, respInd[r]] = zscores[ind[r1]]

    allStimIndsUQ = []
    for i in allStimInds:
        if len(i) > 0:
            allStimIndsUQ.append(i[0])
    allStimIndsUQ = (set(allStimIndsUQ));  # only get the unique stimulated pairs
    print(allStimIndsUQ)

    A = A - np.diag(A)  # make sure there is no activity on the diagonal

    return A, nodeLabels, allStimIndsUQ, chNames

### Function to get peak latency from CCEP response files

# INPUTS:
#  files  = array of the .dat file names
#  peaks = a string, containing the name of the peak of interest to be put
#       into the adjacency matrix- can be n1, p2, n2

# Outputs:
#  chNames = the names of the nodes in the adjacency matrix, as a vector
#       in the same order as they appear in 'L'
#  L = adjacency matrix containing peak latencies of peak of interest
#       specified by input parameter 'peaks'. Stimulating nodes are on the
#       rows, and response nodes are on the columns.
def getPeakLatency(files, peaks):
    # loop over each file containing a stimulation block
    for k in range(len(files)):
        chNames = []
        stimChInd = []
        fileInfo = files[k].split("_")
        stimCh = str(fileInfo[1]) + "_" + str(fileInfo[2]);
        jsonData = json.load(open(files[k]))
        for key in jsonData["zscores"]: chNames.append(key)
        chNames.sort()

        L = np.zeros([len(chNames), len(chNames)])

        stimChInd = getIndxOfOverlap(chNames, stimCh)

        # loop over each response channel and store latency of peak
        for i in range(len(chNames)):
            if jsonData["significant"][chNames[i]]:
                respChInd = getIndxOfOverlap(chNames, chNames[i])
                lVal = jsonData["zscores"][chNames[i]]["n1"][0] - 1 + jsonData["window"][0] * jsonData[
                    "samplingRate"] / 1000 * 1000 / jsonData["samplingRate"]
                L[stimChInd, respChInd] = lVal
            else:
                continue

        return chNames, L

