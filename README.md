# Identification of a Machine Learning-Based Computational Epileptogenic Zone Marker from Cortico-Cortical Evoked Potential Data

## Project Overview
 
Rationale: Epilepsy affects 50 million people globally, and 30-40% of patients have medically refractory epilepsy (MRE), where their seizures cannot be completely controlled by anti-epileptic drugs. A potential cure for MRE is the surgical removal of the epileptogenic zone (EZ), which requires accurate identification of the brain regions that initiate and propagate seizure activity. This remains an unsolved problem, thus surgical success rates currently range from 30-70% of the total epilepsy treatment cost. To localize the EZ, we utilized machine learning methods (ML) to develop a predictive algorithm to identify EZ nodes, leveraging features from cortico-cortical evoked potentials (CCEP) in response to single-pulse electrical stimulation (SPES).
 
Methods: We trained a random forest model based on nodal and network features from CCEP data obtained from 19 patients who underwent intracranial EEG (iEEG) monitoring at Johns Hopkins Hospital. These patients were investigated with single-pulse electrical stimulation (SPES), underwent surgery, and achieved at least 1 year of seizure freedom (Engel class I). Each iEEG contact was considered a node in the CCEPs network that was influenced by the network when neighboring nodes were stimulated, and influential to the network when the node itself was stimulated. Features quantifying influence by and influential to the node were derived from CCEP peak magnitudes, and the node’s importance within the CCEPs network was also captured by graph theory centrality measures, including eigenvector and closeness centrality. These categories of nodal features, splitted into 75:25 train-test sets, were then used to train a random forest model for binary classification whether an iEEG contact was within or outside the EZ. We then tested our model on patients with failed surgical outcomes and were able to indicate potential EZ regions that were not identified initially by conventional clinical analysis.  
 
Results: Our random forest model achieved a separability of EZ sites with an AUC of 87%, accuracy of 78%, sensitivity of 92%, and specificity of 75%. Preliminary comparison of model outcomes and clinical theses on patients with unsuccessful clinical EZ localization has also been promising, with the model predicting alternate locations for the EZ than what was treated.
 
Conclusions: Our study is the first to 1) demonstrate the direct clinical utility of CCEP nodal and network features in EZ localization in a machine-learning framework and 2) apply the extracted CCEP features to a robust ML model (random forest) that may serve as a complementary clinical tool to improve outcomes  after epilepsy surgery. 

## Setup instructions
1) git clone the repo
2) ```cd``` into new ```epilepypcm``` directory
3) do ```python -m pip install -e .``` to install the epilepsypcm repo

## Using the pretrained model
1) The pretrained random forest model can be found [here](final_random_forest.sav)
2) Instructions on how to load this model can be found in [this notebook](Using%20the%20Pretrained%20Random%20Forest%20Model.ipynb)


## Accessing the data
1) Data source: Single pulse electrical stimulation (SPES) recording data in the form of cortico-cortical evoked potentials (CCEPs) was analyzed retrospectively from 44 patients who underwent phase II pre-surgical evaluation at Johns Hopkins Epilepsy center from 2016 to 2021. Details regarding SPES data acquisition and pre-processing are described in detail here: https://doi.org/10.1002/hbm.25418
2) The data from patients with positive surgical outcomes (Engel score of 1) can be found [here](\experiments\final_df.csv) in the experiments folder. All model training and analysis was done on this data, as the surgically resected region from patients with positive surgical outcomes was used as a "ground truth" for training and evaluation.
3) The data from patients with negative surgical outcomes (Engel score of 2-4) can be found [here](\experiments\unsuccesful_df.csv) in the experiments folder. Node that these patients do not have associated "ground truth" labels, as their surgeries were not succesful. The model was trained on these patients to produce nodal predictions - these output files can be found in the unsuccesful_outcome_predictions folder within experiments.

## Analysis
1) Summaries of the random forest model, model-wise comparisons, feature analysis, upsampling analysis, patient-wise clustering, and misc statistical analysis can be found in the experiments folder.

