#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:56:27 2024

@author: marcos
"""

from pathlib import Path

from analysis import load_single_channel, find_valid_periods
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import calc_mode

#%% Graficar un registro y su período en función del tiempo

file = Path('/media/marcos/DATA/marcos/FloClock_data/agus/LS01.abf')

# parámetros de análisis
downsampling_rate = 10
outlier_mode_proportion = 1.8

info_file = pd.read_excel(file.parent / 'info.xlsx').set_index('name')
rec_nr = file.stem
interval = info_file.loc[rec_nr]['rango_de_minutos'] 

# Process data a bit
data = load_single_channel(file, interval)
data.process.lowpass_filter(frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
data.process.highpass_filter(frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
data.process.lowpass_filter(frequency_cutoff=2, keep_og=True, channels='lpfilt_hpfilt')

data.process.find_peaks(channels='lpfilt_hpfilt_lpfilt', period_percent=0.4, prominence=3)

# Plot data and peaks
fig, (ax_raw, ax, ax_p) = plt.subplots(3, 1, figsize=(16, 8), constrained_layout=True, 
                               sharex=True, height_ratios=[2, 2,1])

step = slice(None, None, downsampling_rate//10)
ch = 1

# plot timeseries        
ax_raw.plot(data.times, data[f'ch{ch}'], color='0.6')

ax.plot(data.times, data[f'ch{ch}'] - data.process.get_hptrend(ch), color='0.6')
ax.plot(data.times[step], data[f'ch{ch}_lpfilt_hpfilt'][step])
ax.plot(data.times[step], data[f'ch{ch}_lpfilt_hpfilt_lpfilt'][step])

ax.set_xlim(data.times.values[0], data.times.values[-1])

# plot periods
period_times, periods = data.process.get_periods(ch)
period_mode = calc_mode(periods)   

# plot mode, mean and trend line
valid_period_inxs = find_valid_periods(period_times, periods, outlier_mode_proportion, passes=2)
valid_periods = periods[valid_period_inxs]
period_mean = np.mean(valid_periods)
meanline = ax_p.axhline(period_mean, color='0.3', linestyle=':', zorder=1)

# plot edge crossing periods
rising, multi_rising = data.process.get_multi_crossings(ch, 'rising', 
                                                        threshold=5, threshold_var=5, 
                                                        peak_min_distance=0.4)


# "mrising" for multi-rising edge crossing
mrising_out = data.process.get_multi_edge_periods(ch, 'rising',
                                                threshold=5, threshold_var=5,
                                                peak_min_distance=0.4)
mrising_times, mrising_periods, mrising_errors, mrising_ptp = mrising_out
ax_p.errorbar(mrising_times, mrising_periods, mrising_ptp, fmt='.', color='C0')

# plot mode, mean and trend line for edge periods
valid_period_inxs = find_valid_periods(mrising_times, mrising_periods, outlier_mode_proportion, passes=2)
valid_periods = mrising_periods[valid_period_inxs]
period_mean = np.mean(mrising_periods)
period_mode = calc_mode(mrising_periods) 

# plot average periods on the data
ax_p.plot(mrising_times[~valid_period_inxs], mrising_periods[~valid_period_inxs], 'xr')

fig.suptitle(data.metadata.file.stem)
ax_p.set_xlabel('time (s)')

ax.set_ylabel('mV')
ax_p.set_ylabel('period (s)')

ax.set_title('Channel 1')
ax_p.set_title(f'Channel 1 - bursting period = {period_mean:.2f} sec')

print('Running', data.metadata.file.stem)

#%% Extraer info del registro

file = Path('/media/marcos/DATA/marcos/FloClock_data/agus/LS01.abf')
downsampling_rate = 10
outlier_mode_proportion = 1.8


info_file = pd.read_excel(file.parent / 'info.xlsx').set_index('name')
rec_nr = file.stem
interval = info_file.loc[rec_nr]['rango_de_minutos'] 

# Process data a bit
data = load_single_channel(file, interval)
data.process.lowpass_filter(frequency_cutoff=10, keep_og=True)
data = data.process.downsample()
data.process.highpass_filter(frequency_cutoff=0.1, keep_og=True, channels='lpfilt')
data.process.lowpass_filter(frequency_cutoff=2, keep_og=True, channels='lpfilt_hpfilt')

data.process.find_peaks(channels='lpfilt_hpfilt_lpfilt', period_percent=0.4, prominence=3)

# Plot data and peaks
fig, (ax, ax_p) = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=True, 
                               sharex=True, height_ratios=[2,1])

step = slice(None, None, downsampling_rate//10)
ch = 1

# period from multi-rising edge crossing
mrising_out = data.process.get_multi_edge_periods(ch, 'rising',
                                                threshold=5, threshold_var=5,
                                                peak_min_distance=0.4)
mrising_times, mrising_periods, mrising_errors, mrising_ptp = mrising_out

# get valid periods
valid_period_inxs = find_valid_periods(mrising_times, mrising_periods, outlier_mode_proportion, passes=2)
valid_periods = mrising_periods[valid_period_inxs]
period_mean = np.mean(mrising_periods)

# baseline
baseline, _ = data.process.baseline()

print(data.metadata.file.stem)
print(f'\tperiod = {period_mean:.2f}sec')
print(f'\tbaseline= {baseline:.2f}mV')

