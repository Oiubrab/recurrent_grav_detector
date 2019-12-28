import numpy as np
power_spectral_density = np.loadtxt('ZERO_DET_high_P_PSD_0.25Hz.txt')
print(power_spectral_density)
print(power_spectral_density.shape)
print(power_spectral_density[1,0])
print(power_spectral_density[0,0])
