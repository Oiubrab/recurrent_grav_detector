from __future__ import division
import sys
sys.path.insert(1, '../tools')
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import lalsimulation as lalsim 
from scipy import signal
import scipy.interpolate
import os
import random as ran
from functional_three_glitch import *
from machine_run import *
import shutil
from keras.models import Sequential
#from contextlib import redirect_stdout
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Reshape
from keras.layers import Activation
from keras.utils import plot_model
from utilities_three import *
from sklearn.metrics import confusion_matrix
import time
import subprocess
#from time import process_time
from keras.models import Sequential
from keras.models import model_from_json

# record time at start
t_start = time.time()

# load in the time series construction constants and printout path
srate, f_min, f_max, duration, df, lo, hi, dur_sig, df_sig, lo_sig, hi_sig, N_fd = machine_configuration()
directory = printout()



### Data Generation Stage ###


# initialise data dictionaries
test_train_dic={"train":[0,0,0,0],"test":[0,0,0,0]}
series_dic_list = {"train":[0],"test":[0]}
series_dic_result = {"train":[0],"test":[0]}

# collect arguments from initialisation and make relevant folder
no_arg = len(sys.argv)
if (no_arg == 14):
	test_train_dic["train"][0]=int(sys.argv[1])
	test_train_dic["train"][1]=int(sys.argv[2])
	test_train_dic["train"][2]=int(sys.argv[3])
	test_train_dic["train"][3]=int(sys.argv[4])
	
	test_train_dic["test"][0]=int(sys.argv[5])
	test_train_dic["test"][1]=int(sys.argv[6])
	test_train_dic["test"][2]=int(sys.argv[7])
	test_train_dic["test"][3]=int(sys.argv[8])
	
	gauss_sigma = int(sys.argv[9])
	gauss_amp = int(sys.argv[10])
	test = str(sys.argv[11])
	train = str(sys.argv[12])
	print_out = str(sys.argv[13])
else:
	print("Usage: python scavenger_of_black_hole_signals_seven_combine_non_overlapping_sets_with_glitch_distinguish.py <Data_ID>")
	print("where Data_ID is ten numbers and three instructions")
	print("correspinding: \n\nno of training signals, \nno of training noise, \nfurthest BBH (Mpc) (training), \nclosest BBH (Mpc) (training), \nno of testing signals, \nno of testing noise, \nfurthest BBH (Mpc) (testing), \nclosest BBH (Mpc) (testing), \nsigma of gaussian glitch, \nmaximum amplitude of gaussian glitch, \ntraining data generation format, \ntesting data generation format, \nprint out data to text file (yes/no)")
	print("\nExample: python scavenger_of_black_hole_signals_seven_combine_non_overlapping_sets_with_glitch_distinguish.py 100 100 750 200 50 50 750 200 100 10 integer half yes")
	sys.exit()

# make the folder that will contain all the data 
os.makedirs(directory)

# Make all the data

for test_train in test_train_dic:
	
	# initialise the data set variables for the relevant test/train run
	B_H_signals = test_train_dic[test_train][0]
	N_signals = test_train_dic[test_train][1]
	Ld_higher = test_train_dic[test_train][2]
	Ld_lower = test_train_dic[test_train][3]
	
	# prepare relevant folder for data saving if print out selected
	if print_out=='yes':
		dirName = directory+'/data_series_'+test_train
		if not os.path.exists(dirName):
			os.makedirs(dirName)
		else:	
			shutil.rmtree(dirName)
			os.makedirs(dirName)
	
		print('Directory ' +dirName+  ' Created ')
		print('info dump')
	
	# this is the total number of data sets to generate. 
	total_series=B_H_signals+N_signals
	
	# creating the text files and putting them into the created directory
	# I want to randomise the order of the numerical name attached to each data series
	# but the number of each form (noise or noise+signal) should appear as many times as entered
	
	# create a list of randomised numbers for the random signa/noise generator
	pick=ran.sample(range(total_series),total_series)
	# check if I want static luminosity distance simulations or a range of luminosity distances
	if (Ld_higher==Ld_lower):
		Ld_a=np.full(total_series, Ld_higher)
	else:
		Ld_range=Ld_higher-Ld_lower
		# if using integer/half integer, generate/concatenate multiple lists, so that the luminosity distance array is larger than the total series length
		repetition_ceiling=int(mt.ceil(total_series/Ld_range))
		# create a list of randomised luminosity distances within the range and no overlapping training/ testing data, for testing data only
		if (test == 'integer' and test_train=='test') or (train=='integer' and test_train=='train'):
			Ld_a=ran.sample(range(Ld_lower,Ld_higher), Ld_range)*repetition_ceiling
		elif (test == 'random' and test_train=='test') or (train=='random' and test_train=='train'):
			Ld_a=np.random.uniform(low=Ld_lower, high=Ld_higher, size=total_series)
		elif (test == 'half' and test_train=='test') or (train=='half' and test_train=='train'):
			Ld_a=ran.sample(np.linspace(Ld_lower+0.5,Ld_higher-0.5, num=Ld_range), Ld_range)*repetition_ceiling

	
	#initialise parameter, series classifier and series dictionaries and internal arrays
	p_list=[]
	series_dic_list[test_train] = np.empty([total_series,int(srate)])
	series_dic_result[test_train] = np.empty(total_series)
	
	### we need to read in the LIGO design sensitivity PSD 
	#power_spectral_density = np.loadtxt('/Users/jpowell/projects/bilby/bilby/gw/noise_curves/	aLIGO_ZERO_DET_high_P_psd.txt')	
	power_spectral_density = np.loadtxt('../tools/ZERO_DET_high_P_PSD_0.25Hz.txt')
	
	# this is how long the time series should be
	Nt_noise = int(4096 * duration)
	
	# need to change the sampling of the psd to our required size for the 1s of noise
	psd_interpolant = scipy.interpolate.interp1d(power_spectral_density[:, 0], power_spectral_density[:, 1])
	new_frequency_array = np.arange(f_min, f_max, df)
	new_psd = psd_interpolant(new_frequency_array)
	
	# generate the plots, using the random list to number the choices
	B_H_count=0
	for x in pick:
		
		# create randomised amplitude glitch (gaussian), place randomly 
		glitch = glitch_king(gauss_amp,gauss_sigma,Nt_noise)
		
		# here we get the simulated data
		frequency_domain_strain = GetSimulatedData(new_psd, N_fd, new_psd, df)
		
		# whiten the noise
		time_noise_only = np.fft.irfft(frequency_domain_strain, n=Nt_noise) * srate
		time_noise_whitened = whiten(time_noise_only,new_psd, srate, lo, hi)
		
		if (B_H_count<B_H_signals):
			# list parameters for your signal that you can change
			
			mass_1 = 20
			mass_2 = 20 
			luminosity_distance = Ld_a[x]
			a_1 = 0.0
			tilt_1 = 0.0
			phi_12 = 0.0
			a_2 = 0.0
			tilt_2 = 0.0
			phi_jl = 0.0
			iota = 0.0
			phase = 0.0
			ra = 1.375
			dec = -1.2108
			geocent_time = 1126259642.413 
			psi = 0.0
			# make signal
			hplus, hcross, freqvals = lal_binary_black_hole(mass_1, mass_2, luminosity_distance, a_1, tilt_1, 	phi_12, a_2, tilt_2, phi_jl, iota, phase, ra, dec, geocent_time, psi, srate, dur_sig)
	
			
			# Now we need to calculate the antenna pattern and apply it to our signal.
			# For now I am just going to assume that F+ anf Fx are 0.4 but really it
			# should be calculated for different sky positions and GPS times. We
			# can add code for that later.
			
			Fplus = 1.0
			Fcross = 1.0
			
			freq_domain_signal = hplus*Fplus + hcross*Fcross*1j
			
			# need to change the sampling of the psd to our required size for signal
			# this is smaller than noise to leave some room at the edges of the 1s segment of noise 
			new_psd_sig = psd_interpolant(freqvals[lo_sig:])
			# Now we can try and calculate the SNR for our signal 
			SNR = calculate_snr(freq_domain_signal, new_psd_sig)
			if (test_train=="train" and Ld_higher==Ld_lower):
				SNR_indicate = SNR
			# Now fourier transform to convert to time series
			# I'm going to zero pad it later.
			Nt_signal = int(4096 * dur_sig) 
			
			time_signal_only = np.fft.irfft(freq_domain_signal, n=Nt_signal) * srate
			
			# Make it the same length as the noise by zero padding
			temp = np.zeros([int(srate*duration)])
			sigsize = len(time_signal_only)
			temp[0:sigsize] = time_signal_only
			
			# apply a window
			windowVals = signal.tukey(4096*int(duration), alpha=0.15)
			time_signal_only = temp * windowVals 
		
			# apply the whitening to our signal 
			time_signal_whitened = whiten(time_signal_only,new_psd, srate, lo, hi)
			# go through the construction of noise plus signal and save
			data_with_signal = time_noise_whitened + time_signal_whitened
			# save the masses and the SNR to a 2d array to be written to txt later
			# if working with single model, forget the SNR saving
			if (Ld_lower==Ld_higher):
				p_list=p_list+[[x+1]]
			else:
				p_list=p_list+[[luminosity_distance,SNR,x+1]]
			# and finally, save the noise + signal data into the 2d array
			series_dic_list[test_train][x] = data_with_signal
			series_dic_result[test_train][x] = int(1)
			B_H_count=B_H_count+1
		else:
			# save the construction of noise + glitch
			
			time_noise_whitened += glitch
			series_dic_list[test_train][x] = time_noise_whitened
			series_dic_result[test_train][x] = int(0)
	
	# first, save the series and series classifiers into their relevant folders	if chosen to	
	# then open the relevant meta text file and delete any old ones
	if print_out=='yes':
		np.savetxt(dirName+'/series_data',series_dic_list[test_train])
		np.savetxt(dirName+'/series_result',series_dic_result[test_train])
		if os.path.isfile(dirName+'/meta_data'):
			os.remove('training_meta')
	
	# open the relevant log file
	if print_out=='yes':
		f=open(dirName+'/meta_data_'+test_train,'w+')
	else: 
		f=open(directory+'/meta_data_'+test_train,'w+')

	
	# calculate the optimal snr of the biggest glitch
	maxi_glitch = glitch_king(gauss_amp,gauss_sigma,Nt_noise,maxi=True)
	maxi_glitch_psd, freq_glitch = plt.psd(maxi_glitch)
	white_noise_psd, freq_noise = plt.psd(time_noise_whitened)
	plt.clf()
	glitch_snr = np.mean(maxi_glitch_psd)/np.mean(white_noise_psd)
	
	# save a log of the important variables in the generation process for human reading
	f.write('#Luminosity Distances are between %s and %s megaparsecs \r'%(Ld_lower,Ld_higher))
	f.write('#noise and signal=')
	f.write(str(B_H_count)+'\r')
	f.write('#noise=')
	f.write(str(total_series-B_H_count)+'\r')
	f.write('#total=')
	f.write(str(total_series)+'\r')
	f.write('#max glitch amplitude=')
	f.write(str(gauss_amp)+'\r')
	f.write('#max glitch sigma=')
	f.write(str(gauss_sigma)+'\r')
	f.write('#max glitch snr=')
	f.write(str(glitch_snr)+'\r')
	f.write('#mass 1='+str(mass_1)+'\r')
	f.write('#mass 2='+str(mass_2)+'\r')
	f.write('#a_1='+str(a_1)+'\r')
	f.write('#tilt_1='+str(tilt_1)+'\r')
	f.write('#phi_12='+str(phi_12)+'\r')
	f.write('#a_2='+str(a_2)+'\r')
	f.write('#tilt_2='+str(tilt_2)+'\r')
	f.write('#phi_jl='+str(phi_jl)+'\r')
	f.write('#iota='+str(iota)+'\r')
	f.write('#phase='+str(phase)+'\r')
	f.write('#ra='+str(ra)+'\r')
	f.write('#dec='+str(dec)+'\r')
	f.write('#geocent_time='+str(geocent_time)+'\r')
	f.write('#psi='+str(psi)+'\r\n')
	if (Ld_lower==Ld_higher):
		f.write('#SNR='+str(SNR)+'\r\n')
		f.write('#Luminosity List:\r\n')
		f.write('#Series Number\r')
	else:
		f.write('#Luminosity List:\r\n')
		f.write('#Luminosity              SNR                      Series Number\r')
	np.savetxt(f,p_list)
	f.close()
	




### Network training Stage ###


N_sequence = 4096     # Length of each piece of data
N_epochs = 100       # Number of epochs

# Define confusion matrix labels
class_names=np.array(['no signal','signal'])

# pass the training and testing data into the relevant variables
X_train = series_dic_list["train"]
Y_train = series_dic_result["train"]
X_test = series_dic_list["test"]
Y_test = series_dic_result["test"]

# define training/testing sizes
N_test = len(X_test)
N_train = len(X_train)

print("Training with ",N_train," elements")

# reshape for LSTM model - uncomment when ready

#X_train=X_train.reshape(N_train,N_sequence,1)
#X_test=X_test.reshape(N_test,N_sequence,1)

# Create our Keras model - an RNN (in Keras this is a Sequence)
model = Sequential()

# dense model

model.add(Dense(32, activation='sigmoid',input_dim=N_sequence))
model.add(Dropout(0.1))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='softmax'))
model.add(Dense(1, activation='relu'))

# LSTM model

# short model - uncomment when ready

#model.add(LSTM(32, activation='tanh', recurrent_activation='hard_sigmoid', input_shape=(N_sequence, 1)))
#model.add(Dropout(0.1))
#model.add(Dense(16, activation='softmax'))
#model.add(Dense(1, activation='relu'))

# long model - uncomment when ready

#model.add(LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid', input_shape=(N_sequence, 1)))
#model.add(Dense(256, activation='sigmoid',))
#model.add(Dropout(0.3))
#model.add(Dense(256, activation='softmax'))
#model.add(Dropout(0.1))
#model.add(Dense(64, activation='tanh'))
#model.add(Dropout(0.05))
#model.add(Dense(16, activation='softmax'))
#model.add(Dense(1, activation='relu'))

# Compile model and print summary
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Fit the model using the training set
history = model.fit(X_train, Y_train, validation_split=0.33, epochs=N_epochs, batch_size=32, validation_data=(X_test, Y_test))

# Plot the history
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig(directory+'/figure_acc_%s_%s' % (len(Y_train),len(Y_test)))
# close the accuracy figure
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(directory+'/figure_loss_%s_%s' % (len(Y_train),len(Y_test)))
plt.close()

# Prepare data for confusion matrix
y_pred=model.predict_classes(X_test)
y_test=Y_test.astype(int)

# save normalised confusion matrix to file
wow,confusing=plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion matrix, without normalization')
plt.savefig(directory+'/confusion_matrix_non_normalised_%s_%s' % (len(Y_train),len(Y_test)))
plot_confusion_matrix(y_test, y_pred, class_names, normalize=True, title='Normalized confusion matrix')
plt.savefig(directory+'/confusion_matrix_%s_%s' % (len(Y_train),len(Y_test)))

# Final evaluation of the model using the Test Data
print("Evaluating Test Set")

# define probability metrics
FP=float(confusing[0,1])
FN=float(confusing[1,0])
TP=float(confusing[1,1])
TN=float(confusing[0,0])

# compute sensitivity and accuracy
sensitive=TP/(TP+FN)
accurate=(TP+TN)/(TP+TN+FN+FP)
FAR=FP/(FP+TN)
# print these scores out as percentages
print("Accuracy: %.2f%%" % (accurate*100))
print("Sensitivity: %.2f%%" % (sensitive*100))
print("False Alarm Ratio: %.2f%%" % (FAR*100))

# save luminosity distance, snr, accuracy and sensitivity into text file additively. remove snr if it is trained over a range
try:
	globe=open(directory+'/luminosityDistance_snr_accuracy_sensitivity','a')
	np.savetxt(globe,[[test_train_dic["train"][3],SNR_indicate,accurate,sensitive]])
except NameError:
	subprocess.call(["rm", directory+'/luminosityDistance_snr_accuracy_sensitivity'])
	globe=open(directory+'/luminosityDistance_accuracy_sensitivity','a')
	np.savetxt(globe,[[test_train_dic["train"][2],accurate,sensitive]])

# Export the model to file
model_json = model.to_json()
with open(directory+'/model.json', "w") as json_file:
        json_file.write(model_json)
# Save the weights as well, as a HDF5 format
model.save_weights(directory+'/model.h5')

#record the order of training for reproducibility
if os.path.isfile('training_order'):
		os.remove('training_order')
np.savetxt(directory+'/training_order',Y_train)

#record the order of testing for reproducibility
if os.path.isfile('testing_order'):
		os.remove('testing_order')
np.savetxt(directory+'/testing_order',Y_test)





### Mass Inference Stage ###

# print space from last stage
print(" ")

# We still need to know how long the time series is

N_infer=ran.sample(range(int(len(X_test))),10)    # length of data sets

for i in N_infer:
	
	# pick the specific data line
	X_infer= X_test[i]
	Y_infer=[Y_test[i]]
	X_infer = np.array([X_infer])
	
	# Now try classifying the single data line we loaded
	Class_infer = model.predict_classes(X_infer)
	
	# Compute the class predictions - shouldn't be used as certainties.
	Class_prob = model.predict(X_infer)
	
	print("for test Series %s"%(i+1))
	print("The predicted class is %d" % Class_infer[0])
	print("Class Predictions: Class 0 = %f, Class 1 = %f" % ((1.0-Class_prob[0]), Class_prob[0]))
	print("The actual loaded class is %d" % Y_infer[0])
	print(" ")






# record time at end, calculate elapsed time and print
t_end = time.time()
t_elapsed = t_end - t_start
t_hr = mt.floor(t_elapsed//3600)
t_min = mt.floor((t_elapsed%3600)/60)
t_sec = t_elapsed%60
print("time elapsed = %s hrs, %s mins, %s s" %(t_hr,t_min,t_sec))
