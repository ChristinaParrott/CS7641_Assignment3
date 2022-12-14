Assignment 3: Unsupervised Learning and Dimensionality Reduction
GA Tech CS 7641 - Machine Learning

Author: Christina Parrott
Student ID: 910117883
GT ID: chipps3

CODE IS LOCATED HERE:
    https://github.com/ChristinaParrott/CS7641_Assignment3/
DATASETS ARE LOCATED HERE:
    1. Heart data: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
    2. Weather data: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

CONTENTS OF THIS REPOSITORY:
    FOLDERS
        - datasets: Contains two unaltered datasets downloaded from the following sources:
            1. heart_2020_cleaned.csv: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
            2. weatherAUS.csv: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
        - images: this is the output location for all plots generated by the code. Rerunning the code will overwrite
            the images in this folder.
    FILES
        - environment.yml: conda environment specification.
        - experiments.py: runs experiments for each of the given supervised learning algorithms
        - output.txt: this output file contains any relevant data generated by the code that is subsequently
            referenced in the assignment 3 paper. Successive runs will append to this file.
        - requirements.txt: requirements file designed for use with pip if the grader does not wish to use conda.

INSTRUCTIONS FOR RUNNING:
    OPTION 1- CONDA
        This code may be run in a conda environment using the provided environment.yml file. To do so, follow these steps:
        1. Install conda using the instructions from the following link:
            https://conda.io/projects/conda/en/latest/user-guide/install/index.html
        2. From a command line, navigate to the directory containing the environment.yml file and issue the following command:
            conda env create environment.yml
        3. Run the following command to activate the newly created environment:
            conda activate unsupervised_learning
        4. Finally, you may run the code by issuing the following command:
            python experiments.py
    OPTION 2- REQUIREMENTS FILE
        If you do not wish to utilize the conda environment, you can manually install dependencies by doing the following
        1. This code was written for Python version 3.10, but it should run on newer versions on Python without difficulty.
           If needed, Python 3.10 can be obtained from the following link: https://www.python.org/downloads/release/python-3100/
        2. From a command line, navigate to the directory containing the requirements.txt file and issue the following command:
           pip install -r requirements.txt
        3. Finally, you may run the code by issuing the following command:
            python experiments.py
