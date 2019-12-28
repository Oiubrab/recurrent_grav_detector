# those variables that define the data set generation 
def machine_configuration():
	# some constants
	srate = 4096.0
	f_min = 30.
	f_max = 2048.
	# these ones for the noise
	duration = 1.0
	df = 1.0 / duration
	lo = int(f_min/df)
	hi = int(f_max/df) 
	# these ones for the signal that's shorter than the noise 
	dur_sig = 0.9
	df_sig = 1.0 / dur_sig
	lo_sig = int(f_min/df_sig)
	hi_sig = int(f_max/df_sig) +1
	# size of freq data noise 
	N_fd = (srate * duration / 2.0) - lo # + 1.0
	
	return srate, f_min, f_max, duration, df, lo, hi, dur_sig, df_sig, lo_sig, hi_sig, N_fd

# define the path for all the data to be outputted
def printout():

	direc='../recurrent_grav_sim_data' 

	return direc
