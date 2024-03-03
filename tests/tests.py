#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:16:13 2024

@author: marcos
"""

from spikelib import load_data
import numpy as np

#%% Multiple sweeps

# file = '/home/marcos/Documents/Spikelib/test_data/agus/agus01.abf'
file = '/home/user/Documents/Doctorado/SpikeLib/test_data/agus/agus01.abf'

data = load_data(file, sweep=1)
data.process.downsample(100)

data.process.highpass_filter(keep_og=True)
data.process.poly_detrend(keep_og=True)
# data.process.magnitude_detrend(keep_og=True, column='hpfilt')

ax = data.process.plot('all')
ax.plot(data.times, data.process.get_hptrend())
ax.plot(data.times, data.process.get_polytrend())

#%% Multiple channels

file = '/home/marcos/Documents/Spikelib/test_data/flor/LL03.abf'
data = load_data(file, channel=1)
data.process.downsample(100)

data.process.lowpass_filter(keep_og=True)
btimes, baselines = data.process.calc_multi_baseline(column ='lpfilt', length = 20)

data.process.lowpass_filter(frequency_cutoff=2, keep_og=True)
data.process.find_peaks(column='lpfilt')

crossing_indexes, multicrossing_indexes = data.process.get_multi_crossings('rising', threshold=-40, threshold_var=7)
mci = multicrossing_indexes.flat[~np.isnan(multicrossing_indexes.flat)].astype(int)

ax = data.process.plot('all')
ax.plot(btimes, baselines)

ax.plot(data.process.get_peak_pos(), data.process.get_peak_values(), 'x')
ax.plot(data.times[mci], data.rec_lpfilt[mci], '.k')
ax.plot(data.times[crossing_indexes], data.rec_lpfilt[crossing_indexes], 'o')


#%% No info file

file = '/home/marcos/Documents/Spikelib/test_data/giu1.abf'
data = load_data(file)
data.process.lowpass_filter(frequency_cutoff=20, keep_og=True, column='')
data.process.lowpass_filter(frequency_cutoff=10, keep_og=True, column='')
data.process.lowpass_filter(frequency_cutoff=2, keep_og=True, column='')

downsampled = data.process.downsample(1000, inplace=False)
ax = data.process.plot('all')
downsampled.process.plot(ax=ax)

#%% Voltage clamp (current plots)

file = '/home/marcos/Documents/Spikelib/test_data/Maqui/22o19014 Vclamp con estimulacion con luz.abf'
data = load_data(file, sweep=20, channel=0)
data.process.plot()