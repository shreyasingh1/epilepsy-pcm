#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 20:13:12 2021

@author: richardlee
"""
import numpy as np
import json
from statistics import mean
import os
import glob
import pandas as pd
from epilepsypcm.loading.imports import respInfoToAdjacencyMatrix, zToAdjacencyMatrix, getPeakLatency

base_path = '/Users/richardlee/Desktop/2021 Fall/Precision Care Medicine/Coding/PY16N008'
response_path = base_path + '/ResponseInfo/CCEP'
z_path = base_path + '/z-scores'

response_files = os.listdir(response_path)
response_files_path = glob.glob(response_path + '/*.json', recursive=True)

z_files = os.listdir(z_path)
z_files_path = glob.glob(z_path + '/*.json', recursive=True)

A, nodeLabels, allStimInds, chNames = respInfoToAdjacencyMatrix(response_files_path, "n1", 6)

import seaborn as sns
import matplotlib.pyplot as plt

A_positive = A
A_positive[A_positive<0] = 0

plt.figure(figsize = (15,12))
sns.heatmap(A, cmap = "PiYG", xticklabels = chNames, yticklabels = chNames)