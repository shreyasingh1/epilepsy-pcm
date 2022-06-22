#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 19:16:36 2021

@author: richardlee
"""
import json
import tkinter
import tkinter.filedialog as tkFileDialog
import glob
import os
import numpy as np
import pandas as pd

'''
currdir = os.getcwd()
root = tkinter.Tk()
root.withdraw() #use to hide tkinter window
respInfoPath = tkFileDialog.askdirectory(parent=root, initialdir=currdir, title='Please select a directory')

'''
respInfoPath = '/Users/richardlee/Desktop/2021 Fall/Precision Care Medicine/Coding/PY16N008/ResponseInfo/CCEP'


if len(respInfoPath) > 0:
    print ("You chose %s" % respInfoPath)

respInfoFile_path = glob.glob(respInfoPath + '/*.json', recursive=True)
respInfoFiles = os.listdir(respInfoPath)

stimCh1 = []
stimCh2 = []
n = 0
for i in range(len(respInfoFiles)):
    txt = respInfoFiles[i]
    fileInfo = txt.split("_")
    stimCh1.append(fileInfo[1]); #name of stimulated channel 1
    stimCh2.append(fileInfo[2]); #name of stimulated channel 2
    
    # get whole json data structure
    f = open(respInfoFile_path[i])
    data = json.load(f)
    f.close()
    
    chNames = list(data['time'].keys())
    
    #loop over each channel, and extract average time series and information about the peaks
    if n < 1:
        avgResp = np.empty((len(respInfoFiles),len(chNames),len(data['time'][chNames[0]])))
        significant = np.empty((len(respInfoFiles),len(chNames)))
        n1Zscore = np.empty((len(respInfoFiles),len(chNames)))
        n2Zscore = np.empty((len(respInfoFiles),len(chNames)))
        p2Zscore = np.empty((len(respInfoFiles),len(chNames)))
        n1Latency = np.empty((len(respInfoFiles),len(chNames)))
        n2Latency = np.empty((len(respInfoFiles),len(chNames)))
        p2Latency = np.empty((len(respInfoFiles),len(chNames)))
        flipped = np.empty((len(respInfoFiles),len(chNames)))
        n = n+1
        samplingRate = np.empty((len(respInfoFiles)))
        window = np.empty((len(respInfoFiles),2))
        
        
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
    


