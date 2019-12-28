
# Utilities_three.py
# Dr. Matthew Smith, Swinburne University of Technology
# modified by Adrian Barbuio, Honours student at Swinburne University of Technology
# Various tools prepared for the ADACS Machine Learning workshop 

# Import modules
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from machine_run import *

# define the data path
directory = printout()

def read_training_data(ID, N):
	# Will load files of names series(Z).txt and series_result(z).txt where
	# (Z) is an integer. 
	# Example: Will load series4.txt and series_result4.txt
	# series4.txt will contain N double precision values.
	# series_result4.txt contains a single integer - the class (i.e. classification).
	# INPUT:
	# ID: This is the value of integer (Z) controlling which file to open
	# N: The number of elements contained within each time series
	# Files are in binary, hence precision needs to be provided.
	X = np.zeros(N); 
	Y = np.zeros(1);
	fname = directory+'/data_series_train/series_data'
	print("Loading file " + fname)
	gargantua = np.loadtxt(fname,'double')
	X = gargantua[ID-1]
	# Now for y
	ftame = directory+'data_series_train/series_result'
	pantagruel = np.loadtxt(ftame,'double')
	Y = pantagruel[ID-1]
	
	#If you are going to add noise to this data set, here is a great place to do it.
	# X = X + <NOISE>
	# You can borrow parts from filter_demo.py for the exact syntax.
	# PS: You could also create (def) a new function (def add_noise(X):) - your call.

	return X,Y

def read_test_data(ID, N):
	# Will load files of names series(Z).txt and series_result(z).txt where
	# (Z) is an integer. 
	# Example: Will load series4.txt and series_result4.txt
	# series4.txt will contain N double precision values.
	# series_result4.txt contains a single integer - the class (i.e. classification).
	# INPUT:
	# ID: This is the value of integer (Z) controlling which file to open
	# N: The number of elements contained within each time series
	# Files are in binary, hence precision needs to be provided.
	X = np.zeros(N);
	Y = np.zeros(1);
	fblame = directory+'data_series_test/series_data'
	print("Loading file " + fblame)
	alcyoneus = np.loadtxt(fblame,'double')
	X = alcyoneus[ID-1]
	# Now for y
	fshame = directory+'data_series_test/series_result'
	heracles = np.loadtxt(fshame,'double')
	Y = heracles[ID-1]
	
	return X,Y


def plot_results(ID, X):
	# Use Matplotlib to plot the data for inspection
	# ID: Data ID, only used for placing in the title.
	# X: Data we are plotting.
	fig,ax = plt.subplots()
	ax.plot(X)
	# Give it some labels
	Title  = "Data Set %d" % ID
	ax.set(xlabel='Time Sequence (t)', ylabel='Data X(t)',title=Title)
	plt.show()
	return

def plot_history(history):
	# Use Matplotlib to view the convergence/training history
	fig,ax = plt.subplots()
	ax.plot(history.history['acc'])
	ax.set(xlabel='Epoch',ylabel='Accuracy',title='Accuracy Convergence History')
	plt.show()
	return
	

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm
