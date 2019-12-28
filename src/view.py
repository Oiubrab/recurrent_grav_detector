# view.py
# Written by Dr. Matthew Smith, Swinburne University of Technology
# Load a single data set and plot using matplotlib
# This assumes you have X11 forwarded if you are running remotely.
# Mac O/S users will need XQuartz installed and running.
# Usage: python view.py 2  <enter>
# This will load the time sequence data from ID=2 in the training set.


import sys
import numpy as np
sys.path.insert(1, '../tools')
from utilities_three import *

# Load training data
N_sequence = 4096;    # Length of each piece of data

# Parse the arguments
no_arg = len(sys.argv)
if (no_arg == 3 and (str(sys.argv[1])=='test' or str(sys.argv[1])=='train')):
	plot_ID = int(sys.argv[2])
	testtrain = str(sys.argv[1])
	# Load a specific data file and plot the results
	# choose either the test or train folders
	if (testtrain=='test'):
		X_train, Y_train = read_test_data(plot_ID,N_sequence)
		plot_results(plot_ID,X_train)
	elif (testtrain=='train'):
		X_train, Y_train = read_training_data(plot_ID,N_sequence)
		plot_results(plot_ID,X_train)
else:
	print("Usage: python view.py test/train <Data_ID>")
	print("Data_ID is a number.")
	print("Type test or train at test/train to choose relevant folder.")
	print("Example: python view.py test 2")
	sys.exit()


