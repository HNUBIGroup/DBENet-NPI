*DBENet-NPI*
--------------------------------------------------------------------------------
Predicting ncRNA-protein interactions based on multi-perspective information and dual-branch encoder network.

*Usage*
--------------------------------------------------------------------------------
- For five datasets, you need to run the sample.py file to generate the corresponding sample.txt file ( ../data/sample.py )
- Select the dataset name and run the main.py file to train the model directly.

*Note*
--------------------------------------------------------------------------------
- data project 
  - This project contains the raw sequences and the processed data of datasets used in this study. 
  - sample project contains three coding methods including sequential information, interval information, and physicochemical property coding method.
  - sample.py is the file used to generate the sample.txt file.

- DBENet-NPI
  - The code of multi-perspective information representation is in the data project. 
  - The code of dual-branch encoder network and prediction model are in the model.py file.
    - The code of MCANet is in the interactive.py file.
    - The code of HMCNet is in the independent.py file. 

- main.py is the file used to train the model.
- metric.py consists of the evaluation metrics used in this study.
- model.py consists of the model architecture.
- requirements.txt contains the python package used in this study.
