# recurrent_grav_detector
A deep neural network designed to detect simulated gravitational waves in a realistic LIGO noise profile

This repo contains one of the experiments with neural networks I ran as part of my honours thesis. It runs a recurrent neural
network on KERAS's sequential model platform. The neurons are connected densly, but are made LSTM in a later model (with no 
discernable improvement in detection). To train the network, one needs to first generate the simulated data, defining the black
hole coalescences in the training and testing data sets. Here, the black holes will equally be 20M_sun, rotating optimaly with 
regard to the hypothetical interferometer. In order to generate the data and train the network, cd into the src directory and 
execute:

Usage: python scavenger_of_black_hole_signals_seven_combine_non_overlapping_sets_with_glitch_distinguish.py <Data_ID>
where Data_ID is ten numbers and three instructions
correspinding: 

no of training signals, 
no of training noise, 
furthest BBH (Mpc) (training), 
closest BBH (Mpc) (training), 
no of testing signals, 
no of testing noise, 
furthest BBH (Mpc) (testing), 
closest BBH (Mpc) (testing), 
sigma of gaussian glitch, 
maximum amplitude of gaussian glitch, 
training data generation format, 
testing data generation format, 
print out data to text file (yes/no)

Example: python scavenger_of_black_hole_signals_seven_combine_non_overlapping_sets_with_glitch_distinguish.py 100 100 750 200 50 50 750 200 100 10 integer half yes

running: 

python scavenger_of_black_hole_signals_seven_combine_non_overlapping_sets_with_glitch_distinguish.py

will printout the above instructions to the terminal. The result of the generation and train stage will be a json file 
containing the model and a h5 file containing the weights of the trained network, saved in a folder,
'recurrent_grav_sim_data'. This folder also contains diagnostic and confusion plots.

If the print out data option was chosen, folders with the training and testing set data (in txt files) will be generated

Bash scripts in the test folder will generate automated test runs of the program across a range of variables and save them
to a local Documents folder under 'experiment_recurrent_gravitational'. In order to run these, cd into the test directory
and run:

chmod u+x data_load_dense_data_range
./data_load_dense_data_range

for a range of tests over different training/testing set sizes, or:

chmod u+x data_load_dense_lum_range
./data_load_dense_lum_range

for a range of tests over a luminosity distance range.
