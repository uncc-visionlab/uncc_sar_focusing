import scipy
import numpy as np
import optuna
# import torch
from PIL import Image
import matplotlib.pyplot as plt
from focusalg_BP import focusalg_BP
from dft_error_func import error_func

# Any important variables
CLIGHT = 299792458.0
PI = 3.141592653589793

# Load the data
mat = scipy.io.loadmat('data_3dsar_pass1_az001_HH.mat')

# Parse the data for easy access
sampleData = mat["data"]["fp"][0][0]
freq = mat["data"]["freq"][0][0]
ant_x = mat["data"]["x"][0][0]
ant_y = mat["data"]["y"][0][0]
ant_z = mat["data"]["z"][0][0]
slant_range = mat["data"]["r0"][0][0]
th = mat["data"]["th"][0][0]
phi = mat["data"]["phi"][0][0]
af_r_correct = mat["data"]["af"][0][0][0][0][0]
af_ph_correct = mat["data"]["af"][0][0][0][0][1]

# Need for getting data values
#   sampleData
#   numRangeSamples
#   numAzimuthSamples
#   delta_x_m_per_pix
#   delta_y_m_per_pix
#   left
#   bottom
#   rmin
#   rmax
#   Ant_x
#   Ant_y
#   Ant_z
#   slant_range
#   startF
#   sar_image_params
#   #   N_x_pix
#   #   N_y_pix
#   #   N_fft
#   #   max_Wy_m
#   #   dyn_range_dB
#   #   Wx_m*
#   #   Wy_m*
#   #   x0_m*
#   #   y0_m*
#   range_vec
#
# *Not needed for focusing image

numRangeSamples = sampleData.shape[0]
numAzimuthSamples = sampleData.shape[1]
ant_Az = np.arctan2(ant_y, ant_x)
startF = np.min(freq)
ant_deltaAz = np.diff(ant_Az)
mean_Ant_deltaAz = np.mean(ant_deltaAz)
deltaF = np.diff(freq, axis=0)

mean_startF = np.mean(startF)
mean_deltaF = np.mean(deltaF)
Wx_m = CLIGHT / (2.0 * np.abs(mean_Ant_deltaAz) * mean_startF)
Wy_m = CLIGHT / (2.0 * mean_deltaF)
N_y_pix = numRangeSamples
N_x_pix = int((Wx_m * N_y_pix) / Wy_m)
x0_m = 0
y0_m = 0
dyn_range_dB = 70
N_fft = numRangeSamples


delta_x_m_per_pix = Wx_m / (N_x_pix - 1)
delta_y_m_per_pix = Wy_m / (N_y_pix - 1)
left = x0_m - Wx_m / 2
bottom = y0_m - Wy_m / 2
rmin = np.min(slant_range)
rmax = np.max(slant_range)

#ifft, fftnorm, fftshift
# sampleData = np.fft.ifft(sampleData, N_fft, axis=0) # Default norm factor is "backwards" which is 1/N
# sampleData = np.fft.fftshift(sampleData,axes=0)

#focus image, use focusalg_BP.m as a template for now 

temp_x = np.arange(0,numAzimuthSamples)
p_x = np.polynomial.Polynomial.fit(temp_x, ant_x[0], deg=2)
p_y = np.polynomial.Polynomial.fit(temp_x, ant_y[0], deg=2)
p_z = np.polynomial.Polynomial.fit(temp_x, ant_z[0], deg=2)

p_x = p_x.convert().coef
p_y = p_y.convert().coef
p_z = p_z.convert().coef

ant_x_true = ant_x.copy()
ant_y_true = ant_y.copy()
ant_z_true = ant_z.copy()

def objective(trial):
    x_vec = np.linspace(x0_m - Wx_m / 2, x0_m + Wx_m / 2, N_x_pix)
    y_vec = np.linspace(y0_m - Wy_m / 2, y0_m + Wy_m / 2, N_y_pix)
    [x_mat, y_mat] = np.meshgrid(x_vec, y_vec)
    z_mat = np.zeros(x_mat.shape)

    p_x0 = trial.suggest_float('p_x0', p_x[0], p_x[0])
    p_x1 = trial.suggest_float('p_x1', p_x[1] - np.abs(p_x[1]) * 0.1, p_x[1] + np.abs(p_x[1]) * 0.1)
    p_x2 = trial.suggest_float('p_x2', p_x[2] - np.abs(p_x[2]) * 0.1, p_x[2] + np.abs(p_x[2]) * 0.1)
    
    p_y0 = trial.suggest_float('p_y0', p_y[0], p_y[0])
    p_y1 = trial.suggest_float('p_y1', p_y[1] - np.abs(p_y[1]) * 0.1, p_y[1] + np.abs(p_y[1]) * 0.1)
    p_y2 = trial.suggest_float('p_y2', p_y[2] - np.abs(p_y[2]) * 0.1, p_y[2] + np.abs(p_y[2]) * 0.1)

    p_z0 = trial.suggest_float('p_z0', p_z[0], p_z[0])
    p_z1 = trial.suggest_float('p_z1', p_z[1] - np.abs(p_z[1]) * 0.1, p_z[1] + np.abs(p_z[1]) * 0.1)
    p_z2 = trial.suggest_float('p_z2', p_z[2] - np.abs(p_z[2]) * 0.1, p_z[2] + np.abs(p_z[2]) * 0.1)

    vel = trial.suggest_float('vel', 0.9, 1.1)

    ant_x = p_x0 + p_x1 * temp_x * vel + p_x2 * temp_x ** 2
    ant_y = p_y0 + p_y1 * temp_x * vel + p_y2 * temp_x ** 2
    ant_z = p_z0 + p_z1 * temp_x * vel + p_z2 * temp_x ** 2

    ant_x = np.expand_dims(ant_x, 0)
    ant_y = np.expand_dims(ant_y, 0)
    ant_z = np.expand_dims(ant_z, 0)

    im_final = focusalg_BP(x_mat, y_mat, z_mat, sampleData, 0, N_fft, ant_x, ant_y, ant_z, np.min(freq), Wy_m, slant_range)
    for pulseIndex in np.arange(1, numAzimuthSamples):
        im_final += focusalg_BP(x_mat, y_mat, z_mat, sampleData, pulseIndex, N_fft, ant_x, ant_y, ant_z, np.min(freq), Wy_m, slant_range)
    Iout = (255 / dyn_range_dB) * ((20 * np.log10(np.abs(im_final)/np.max(np.max(np.abs(im_final))))) + dyn_range_dB)
    Iout[Iout < 0] = 0
    return -error_func(Iout)
    # return -np.mean(scipy.stats.entropy(Iout, axis=0))

study = optuna.create_study()
study.optimize(objective, n_trials=10000, n_jobs=4, gc_after_trial=True)

best_params = study.best_params
ant_x = p_x[0] + best_params["p_x1"] * temp_x + best_params["p_x2"] * temp_x ** 2
ant_y = p_y[0] + best_params["p_y1"] * temp_x + best_params["p_y2"] * temp_x ** 2
ant_z = p_z[0] + best_params["p_z1"] * temp_x + best_params["p_z2"] * temp_x ** 2

# ant_x = p_x[0] + p_x[1] * temp_x + p_x[2] * temp_x ** 2
# ant_y = p_y[0] + p_y[1] * temp_x + p_y[2] * temp_x ** 2
# ant_z = p_z[0] + p_z[1] * temp_x + p_z[2] * temp_x ** 2

ant_x = np.expand_dims(ant_x, 0)
ant_y = np.expand_dims(ant_y, 0)
ant_z = np.expand_dims(ant_z, 0)

x_vec = np.linspace(x0_m - Wx_m / 2, x0_m + Wx_m / 2, N_x_pix)
y_vec = np.linspace(y0_m - Wy_m / 2, y0_m + Wy_m / 2, N_y_pix)
[x_mat, y_mat] = np.meshgrid(x_vec, y_vec)
z_mat = np.zeros(x_mat.shape)
im_final = focusalg_BP(x_mat, y_mat, z_mat, sampleData, 0, N_fft, ant_x, ant_y, ant_z, np.min(freq), Wy_m, slant_range)
for pulseIndex in np.arange(1, numAzimuthSamples):
    im_final += focusalg_BP(x_mat, y_mat, z_mat, sampleData, pulseIndex, N_fft, ant_x, ant_y, ant_z, np.min(freq), Wy_m, slant_range)

#show image
Iout = (255 / dyn_range_dB) * ((20 * np.log10(np.abs(im_final)/np.max(np.max(np.abs(im_final))))) + dyn_range_dB)
Iout[Iout < 0] = 0
im = Image.fromarray(Iout)
im = im.convert("L")
im.save("sar_image.bmp", "BMP")
Iout_fft = np.abs(np.fft.fft2(Iout))
Iout_fft = np.roll(Iout_fft, int(np.round(N_x_pix/2)), 1)
Iout_fft = np.roll(Iout_fft, int(np.round(N_y_pix/2)), 0)
Iout_fft = 20*np.log10(Iout_fft+1)
im_fft = Image.fromarray(Iout_fft)
im_fft = im_fft.convert("L")
im_fft.save("sar_image.fft.bmp","BMP")
# plt.imshow(Iout)
# plt.show()