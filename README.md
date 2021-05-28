Prerequisites:
Before working on this project please make sure that following things should be available and installed.
Hardware Requirements:
-	RAM: min 8GB to 16Gb
-	Memory: 10 GB to any.
-	GPU : Google colab Tesla T4 or NVIDIA GeForce or OZstar Supercomputer GPU.
-	Processor: CPU processor would be fine to upload the data and code to the GPUs
Software Requirements:
-	Annaconda installed with Jupyter Notebook
Make sure you’ve installed all the updated versions of python, Pandas, Numpy, Matplot, 

Detailed Instructions:
This path contains the Jupyter Notebook for predicting heart arrhythmias using deep learning using the dataset from MIH-BIH Arrythmia dataset from https://physionet.org/content/mitdb/1.0.0/ . (Download it and keep it in folder where you wanted to work because this is the dataset which is used though out the whole project)

1) If you’re willing to work on the google colab use the following process:
 you need to use mount the drive into the specific drive.
To achieve it you can use:

from google.colab import drive
drive.mount('/content/drive')

Change your runtime to GPU in notebook setting such that you can run your code under GPU provided by google.
 Your mount drive will look like this. Where you’ll store the data set in the folder. If you’re not sure about the data path the go into the drive folder where it’ll have your PC connected data. Check for the dataset folder and right click on it you’ll get the path and use that data path.
2) First we’ve to check which GPU has been allocated to the project from google. To check that type: !nvidia-smi  You may get the output like in the given picture.
  

To train the model in the GPU first thing we need to do is. Install the GPU.
-	!pip install tensorflow-gpu
After the GPU is installed Import all the libraries to the python file and load the dataset from the drive.

*If you have your own GPU or if you wanted to work on OzStar then you can use the local path details into the data path.

3) In order to work on the wave form database we need to install WFDB package to our python file.
Basic Introduction: The native Python waveform-database (WFDB) package. A library of tools for reading, writing, and processing WFDB signals and annotations. Core components of this package are based on the original WFDB specifications. This package does not contain the exact same functionality as the original WFDB package. It aims to implement as many of its core features as possible, with user-friendly APIs. Additional useful physiological signal-processing tools are added over time.
Installation:
The distribution is hosted on pypi at: https://pypi.python.org/pypi/wfdb/ . To directly install the package from pypi without needing to explicitly download content, run from your terminal:
$ pip install wfdb
Load all the annotations into the WFDB package.
If you wanted to check the check the single patient signal graph you can plot using matplot library.

4) Before working with the tensorflow, install and import the library. Also install Keras tunar using:
-	pip install -q -U keras-tuner

Import all the hyperparameters that are required in the model.

5) Perform all the models that you need to analyse. (DNN, 1DCNN, LSTM) as these models will take much longer than expected please be patient enough to run the models.
Results will be produced accordingly make a note of every model.

You can also analyse the model using the AUC curve. In the end with the accuracy results of the models.

