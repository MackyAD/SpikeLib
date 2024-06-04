#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:23:45 2024

@author: marcos
"""

# from pathlib import Path
import re
import numbers
from pathlib import Path
from functools import partial

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import matplotlib.pyplot as plt

import pyabf

from utils import sort_by, enzip, calc_mode, cprint, find_point_by_value


#%% Classes


##############
### Build a container to save run information 
##############

@pd.api.extensions.register_dataframe_accessor("metadata")
class RunInfo:
    """ This should add a few properties to the pandas.DataFrame """

    def __init__(self, pandas_obj):
        self.file = None
        self.info = None
            
    @property
    def file(self):
        return self._file
    @file.setter
    def file(self, filename):
        if filename is not None and not isinstance(filename, Path):
            filename = Path(filename)
        self._file = filename
    
    @property
    def info(self):
        return self._info
    @info.setter
    def info(self, info_line):
        self._info = info_line
        self._attr_translation_keys = {}
        # in case there was no available spreadsheet
        if info_line is None:
            return
        
        # extract the entries in the info line
        # save the original spreadsheet headers
        for name, value in info_line.items():
            # sanitize headers, save original
            attr_name = re.sub(r'\W+|^(?=\d)','_', name)
            self._attr_translation_keys[attr_name] = name
            
            setattr(self, attr_name, value)
    
    _properties_titles = {
        'channel_count' : 'channel count',
        'channel' : 'channel',
        'sweep_count' : 'sweep count',
        'sweep' : 'sweep nr',
        'sweep_duration_sec' : 'sweep length (sec)',
        'duration' : 'recording length (sec)',
        'rec_datetime' : 'recording date',
        'protocol' : 'protocol',
        'raw_sampling_rate' : 'original sampling rate (Hz)',
        'sampling_rate' : 'sampling rate (Hz)',
        'file' : 'file'
        }
    
    @property
    def contents(self):
        """ Returns the names of all the user-facing attributes of the class"""
        return *self._attr_translation_keys, *self._properties_titles
        
    def __repr__(self):
                
        ## Compile the list of things to display
        info_values = {v:getattr(self, k) for k, v in self._attr_translation_keys.items()}
        properties_values = {k:getattr(self, k) for k in self._properties_titles if hasattr(self, k) }
        
        ## Some special cases
        # round sampling rate if appropiate
        if self.sampling_rate == int(self.sampling_rate):
            properties_values['sampling_rate'] = int(self.sampling_rate)
        # ignore raw sampling rate if it's the same as sampling rate (no downsampling was performed)
        if self.sampling_rate == self.raw_sampling_rate:
            del properties_values['raw_sampling_rate']
        # if only one sweep was availble, don't give more sweep info
        if self.sweep_count == 1:
            del properties_values['sweep']
            del properties_values['sweep_duration_sec']
        # if sweep was unspecified ignore, if not, don't show total length
        if self.sweep is None:
            if 'sweep' in properties_values: del properties_values['sweep']
        else:
            del properties_values['duration']
        # if only one channel is available, don't say which one we use
        if self.channel_count == 1:
            del properties_values['channel_count']
            del properties_values['channel']
        
        ## Build the string
        string = '\n'.join(f'{k}: {v}' for k, v in info_values.items())
        if info_values:
            string += '\n\n'
        string += '\n'.join(f'{self._properties_titles[k]}: {v}' for k, v in properties_values.items())
        return string

##############
### Define a class decorator to register methods in different categories
##############

def category(name: str, /):
    def decorator(func):
        try:
            func._categories.add(name)
        except AttributeError:
            func._categories = {name}
        return func
    return decorator


class Category(set):
    def __repr__(self):
        string = '\n'.join(self)
        return string

def register_categories(cls):
    categories: dict[str, set[str]] = {}
    # Get all methods and their categories
    for name, method in cls.__dict__.items():
        if hasattr(method, '_categories'):
            for category in method._categories:
                categories.setdefault(category, Category()).add(name)

    # Create the category attributes from the aggregated categories
    for category, methods in categories.items():
        setattr(cls, category, methods)

    return cls

filtering = category('filtering')
detrending = category('detrending')
event_finding = category('event_finding')
characteristics = category('characteristics')

##############
### Define analysis and processing functions
##############

@pd.api.extensions.register_dataframe_accessor("process")
@register_categories
class Processors:
    """ Makes the process functions methods of the dataframe. All processors 
    will add an entry into self.process.info stating what they did and 
    what parameters were used. 
    All processors have an optional argument to keep the original values or
    override them. If kept, a new column will be added, appending the processing
    type to the column name.
    All processors have an optional argument to be applied to a given column, 
    rather than the default ch1 and ch2. To select a different column, insert
    into 'columns' the name after ch1_ or ch2_. For example, 'det' will target
    channels 'ch1_det' and 'ch2_det'.
    """
    
    ######################
    ### INITIALIZATION ###
    ######################
    
    _all_info_attributes = '_info', '_polytrend', '_peaks'
    _non_data_columns = 'times', 'pert'
    
    def __init__(self, pandas_obj):
        # save the dataframe as an object to later access
        self._df = pandas_obj
        
        self._init_info_attributes()
    
    def _init_info_attributes(self):
        """Initialize the values of the info attributes as None"""
        for attr in self._all_info_attributes:
            setattr(self, attr, None)
            
    ### Properties and cheap lookups/calculations
    @property
    def info(self):
        """All the processing steps taken and the parameters used."""
        return self._info
    @info.setter
    def info(self, value):
        if self.info is None:
            self._info = (value, )
        else:
            self._info = (*self._info, value)
            
    @property
    def steps(self):
        """Names of the processing steps taken."""
        return tuple( step['step'] for step in self.info )
    
    @property
    def polytrend(self):
        """ The polynomial trend polynome."""
        return self._polytrend
    @polytrend.setter
    def polytrend(self, trend):
        if self.polytrend is not None:
            print('A detrending job was already done. Overriding the previous one.')
        self._polytrend = trend
    
    @characteristics 
    def get_polytrend(self):
        """ Evaluates the polynomial trend."""
        assert 'pdetrend' in self.steps, 'You must run a detrending job first'
        return self.polytrend(self._df.times)
    
    @characteristics 
    def get_hptrend(self):
        assert 'hpfilt' in self.steps, 'You must run a detrending job first'
        
        # check if data was kept after the filter
        step_info = self.get_step_info('hpfilt')
        if not step_info['keep_og']:
            raise ValueError("The data was not kept after filtering, so we can't calculate the trend")
        
        # calculate the trend as the difference between before and after applying the filter
        applied_on = step_info['column']
        prefilt = self._get_column(applied_on) # channel data on which the filter was applied
        posfilt = self._get_column(applied_on+'_hpfilt') # channel data of the applied filter
        return prefilt-posfilt
 
        
    def downsample(self, downsampling_rate=10, inplace=True):
        """
        Downsamples the data by skipping datapoints. 

        Parameters
        ----------
        downsampling_rate : int, optional
            How many points to skip when downsampling. The default is 10.

        Returns
        -------

        """
        action_name = 'downsampling'
        data = self._df
        
        # downsample
        downsampled = data[::downsampling_rate].copy()
        
        if inplace:
            # overwrite old dataframe
            empty = np.empty_like(data.times.values)
            for col_name in downsampled.columns:
                col = downsampled[col_name]
                empty[:col.size] = col
                
                data[col_name] = empty
            
            # cut now extra values
            data.drop(index=list(range(col.size, len(data))), inplace=True)
            
            # record what we did and update sampling rate
            self._add_process_entry(action_name, downsampling_rate=downsampling_rate, inplace=inplace)
            data.metadata.sampling_rate = data.metadata.sampling_rate / downsampling_rate
        
        else:
            # copy metadata and attributes
            for attr in self._all_info_attributes:
                setattr(downsampled.process, attr, getattr(self, attr))
            downsampled.metadata = data.metadata
            
            # record what we did and update the sampling rate
            downsampled.process._add_process_entry(action_name, downsampling_rate=downsampling_rate, inplace=inplace)
            downsampled.metadata.duration = data.times.values.ptp()
            
            return downsampled
                
    def cut(self, start, end, units='index', inplace=True):
        """ Cut the data in the given interval [start, end]. Units can be 
        either 'index' or 'time'. In the former case, start and end are assumed
        to be the int indexes at which the cut happens. In the latter, they are 
        assumed to be the times, and the appropiate indexes are found. """
        
        assert units in ('index', 'time')
        
        action_name = 'cut'
        data = self._df
        
        # get indexes, if required
        if units == 'time':
            
            start = find_point_by_value(data.times.values, start)
            end = find_point_by_value(data.times.values, end)
            
        cut = data[start:end].reset_index(drop=True)
        
        if inplace:
            # overwrite old dataframe
            empty = np.empty_like(data.times.values)
            for col_name in cut.columns:
                col = cut[col_name]
                empty[:col.size] = col
                
                data[col_name] = empty
                
            # cut now extra values
            data.drop(index=list(range(col.size, len(data))), inplace=True)
            
            # record what we did and update sampling rate
            self._add_process_entry(action_name, start=start, end=end, units=units, inplace=inplace)
            data.metadata.duration = data.times.values.ptp()
            
        else:
            # copy metadata and attributes
            for attr in self._all_info_attributes:
                setattr(cut.process, attr, getattr(self, attr))
            cut.metadata = data.metadata
            
            # record what we did and update the sampling rate
            cut.process._add_process_entry(action_name, start=start, end=end, units=units, inplace=inplace)
            cut.metadata.duration = data.times.values.ptp()
            
            return cut

    def itersweeps(self, out: str='data'):
        """
        Make an iterator to iterate over the sweeps of the run. Either yield the
        sliced run or a slice to manually use.

        Parameters
        ----------
        out : str, optional
            If 'data', yield sliced data. If 'slices', yield slices to use 
            manually. Raise ValueError otherwise. The default is 'data'.

        Yields
        ------
        Pandas.DataFrame or slice
        
        NOTE
        ----
        If out=='data', the sliced data bits are slices of a DataFrame that is
        not instantiated with all the metadata of the original data. That can
        still be manually accessed through the original's data metadata. To be
        able to run processor methods on the sliced bits, the user should use
        out='slices' and slice the data manually using Processors.cut by 
        accessing slice.start and slice.stop.
        
        """
    
        # extract some parameters
        data = self._df
        dur = data.metadata.sweep_duration_sec
        count = data.metadata.sweep_count
        
        # if only one weeps was loaded, use count=1
        if data.metadata.sweep is not None:
            count = 1
            
        # make a generator that yields slices
        def slicer_gen():
            for i in range(count):
                start = find_point_by_value(data.times.values, i*dur)
                if i<(count-1):
                    end = find_point_by_value(data.times.values, (i+1)*dur)
                else:
                    end = find_point_by_value(data.times.values, (i+1)*dur) + 1
                 
                yield slice(start, end)
        
        # instantiate the genrator
        slicer = slicer_gen()

        # return either sliced data or slicer itself
        if out=='data':
            for s in slicer:
                yield data.iloc[s]
        elif out=='slices':
            yield from slicer
        else:
            raise ValueError('Out has to be either "data" or "slices"')
            
    
    ##########################
    ### DETRENDING OPTIONS ###
    ##########################
    
    @detrending
    def poly_detrend(self, degree=5, keep_og=False, column=''):
        """
        Use a polynomial to detrend the data.

        Parameters
        ----------
        degree : int, optional
            Degree of the polynomial used to fit the data. Default is 5.
        keep_og : Bool, optional
            Whether to keep the original column or overwrite it. The default is
            False.
        column : str, optional
            A string describing what column to apply the funciton on. The 
            default is '', meaning the raw channel.

        Returns
        -------
        None.

        """
        
        action_name = 'pdetrend'
        data = self._df
        
        t = data.times
        col = self._get_column(column)
        P = np.polynomial.Polynomial
        
        # some filter methods leave behind nans in the data, that raises a LinAlgError
        nan_locs = np.isnan(col)
        
        # fit the data
        trend_poly = P.fit(t[~nan_locs], col[~nan_locs], degree)
        
        # remove the trend from the data, this reintroduces the nans
        y_dtr = col - trend_poly(t)
        
        # save processed data
        self._save_processed_data(y_dtr, keep_og, column, action_name)
        self._add_process_entry(action_name, degree=degree, keep_og=keep_og, column=column)
        self.polytrend = trend_poly
    
    @detrending
    def magnitude_detrend(self, keep_og=False, column=''):
        """
        Use the magnitude calculated through a Hilbert transform to "detrend" a
        signal by dividing it by its magnitude (envelope).

        Parameters
        ----------
        keep_og : Bool, optional
            Whether to keep the original column or overwrite it. The default is
            False.
        column : str, optional
            The column to apply the detrend to. You need to have calculated 
            the phase and magnitude already. If not, this function will do it. 
            Note the limitations of this calculation in magnitude_and_phase.
            The default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'mdetrend'
        
        if not self._check_if_action_was_performed_on_column('hilbert', column):
            self.magnitude_and_phase(column)
        
        col = self._get_column(column)
        mag = self._get_column(column + '_magnitude')
        
        y_dtr = col / mag
        
        self._save_processed_data(y_dtr, keep_og, column, action_name)
        self._add_process_entry(action_name, keep_og=keep_og, column=column)


    ###############
    ### FILTERS ###
    ###############

    @filtering
    def gaussian_filter(self, sigma_ms=20, border_effects=True, keep_og=False, column=''):
        """
        Uses a gaussian kernel to filter data. This gives a result that is 
        absolutely equivalent to the built-in abf gaussian filter, but runs a 
        bit slower. Use this if you no longer have the abf object.

        Parameters
        ----------
        sigma_ms : float, optional
            Sigma in units of milliseconds. The default is 20.
        border_effects: Bool, optional
            Decides whether to keep border effects of mirror the data to remove,
            them at the expense of increased cost. See no_border_effects_call.
            The default is True, keeping border effects.
        keep_og : Bool, optional
            Whether to keep the original column or overwrite it. The default is
            False.
        column : str, optional
            A string describing what column to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """                
        action_name = 'gfilt2'
        data = self._df
        col = self._get_column(column)
        
        # some filter methods leave behind nans in the data, that raises a LinAlgError
        nan_locs = np.isnan(col)
        
        # calculate the sigma in untis of datapoints
        sampling_rate = data.metadata.sampling_rate
        sigma_points = (sigma_ms / 1000) * sampling_rate
        
        # col_filt = gaussian_filter1d(col[~nan_locs], sigma_points)
        col_filt = self._no_border_effect_call(
            partial(gaussian_filter1d, sigma=sigma_points),
            col[~nan_locs],
            border_effects
            )
        
        self._save_processed_data(col_filt, keep_og, column, action_name)
        self._add_process_entry(action_name, sigma_ms=sigma_ms, keep_og=keep_og, column=column)        
    
    @filtering
    def lowpass_filter(self, filter_order=2, frequency_cutoff=10, border_effects=True, keep_og=False, column=''):
        """
        Filter the data in channels using a lowpass butterworth filter. The order
        and frequency cutoff value can be set. It uses a forwards and a 
        backwards pass of the filter, resulting in an effective filter order
        that is twice filter_order.

        Parameters
        ----------
        filter_order : int, optional
            Order of the filter. The default is 2.
        frequency_cutoff : float, optional
            Frequency at which the filter drops by 3dB. The default is 10Hz.
        border_effects: Bool, optional
            Decides whether to keep border effects of mirror the data to remove,
            them at the expense of increased cost. See no_border_effects_call.
            The default is True, keeping border effects.
        keep_og : Bool, optional
            Whether to keep the original column or overwrite it. The default is
            False.
        column : str, optional
            A string describing what column to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'lpfilt'
        data = self._df
        col = self._get_column(column)
        
        # some filter methods leave behind nans in the data, that raises a LinAlgError
        nan_locs = np.isnan(col)
                
        sampling_rate = data.metadata.sampling_rate
        sos = signal.butter(filter_order, frequency_cutoff, btype='lowpass', output='sos', fs=sampling_rate)
        col_filt = self._no_border_effect_call(
            partial(signal.sosfiltfilt, sos),
            col[~nan_locs],
            border_effects
            )
        
        self._save_processed_data(col_filt, keep_og, column, action_name)
        self._add_process_entry(action_name, filter_order=filter_order, frequency_cutoff=frequency_cutoff, border_effects=border_effects, keep_og=keep_og, column=column)  
        
    @detrending
    @filtering
    def highpass_filter(self, filter_order=2, frequency_cutoff=0.1, border_effects=False, keep_og=False, column=''):
        """
        Filter the data in channels using a highpass butterworth filter. The 
        order and frequency cutoff value can be set. It uses a forwards and a 
        backwards pass of the filter, resulting in an effective filter order
        that is twice filter_order.

        Parameters
        ----------
        filter_order : int, optional
            Order of the filter. The default is 2.
        frequency_cutoff : float, optional
            Frequency at which the filter drops by 3 dB. The default is 0.1Hz.
        border_effects: Bool, optional
            Decides whether to keep border effects of mirror the data to remove,
            them at the expense of increased cost. See no_border_effects_call.
            The default is False, removing border effects.
        keep_og : Bool, optional
            Whether to keep the original column or overwrite it. The default is
            False.
        column : str, optional
            A string describing what column to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'hpfilt'
        data = self._df
        col = self._get_column(column)
        
        # some filter methods leave behind nans in the data, that raises a LinAlgError
        nan_locs = np.isnan(col)
                
        sampling_rate = data.metadata.sampling_rate
        sos = signal.butter(filter_order, frequency_cutoff, btype='highpass', output='sos', fs=sampling_rate)
        col_filt = self._no_border_effect_call(
            partial(signal.sosfiltfilt, sos),
            col[~nan_locs],
            border_effects
            )
        
        self._save_processed_data(col_filt, keep_og, column, action_name)
        self._add_process_entry(action_name, filter_order=filter_order, frequency_cutoff=frequency_cutoff, border_effects=border_effects, keep_og=keep_og, column=column)  
        
    @staticmethod
    def _no_border_effect_call(func, data, border_effects):
        """ Call a function (usually a filter) over the data. Avoid border 
        effects if border_effects=False. To do so, before applying func, add a 
        mirrored copy of the data at the end and then a copy of the whole thing
        at the beginning, mirrored too. This means the the total length of the 
        array to which func is applied is four times as big, potentially 
        increasing the cost of the call.
        
        This function will run func(data), so any extra arguments should be
        pre-passed to the relevant function, potentially using `partial`.
        """
        
        if border_effects:
            return func(data)
        else:
            dataatad = np.concatenate((data, data[::-1]))
            dataataddataatad = np.concatenate((dataatad, dataatad))
            res = func(dataataddataatad)
            
            return res[data.size*2:-data.size]
    
    ###############################
    ### ATTRIBUTES CALCULATIONS ###
    ###############################
    
    @characteristics 
    def magnitude_and_phase(self, column=''):
        """
        Calculates the phase and magnitude of the timeseries in channels using 
        the hilbert transform. Assumes the data is already detrended, with mean
        0.

        Parameters
        ----------
        column : str, optional
            A string describing what column to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'hilbert'
        col = self._get_column(column)
        
        # do the calculation
        x = col
        nan_locs = np.isnan(x)
        
        analytic = signal.hilbert(x[~nan_locs])
        phases_short = np.unwrap(np.angle(analytic))
        magnitudes_short = np.absolute((analytic))
        
        magnitudes = np.full(x.shape, np.nan)
        phases = np.full(x.shape, np.nan)
        
        magnitudes[~nan_locs] = magnitudes_short
        phases[~nan_locs] = phases_short
        
        self._save_processed_data(phases, keep_og=True, column=column, action='phase')
        self._save_processed_data(magnitudes, keep_og=True, column=column, action='magnitude')
        self._add_process_entry(action_name, column=column)  
        
    @characteristics   
    def calc_baseline(self, column='', drop_quantile=0.5):
        """
        Uses calc_baseline_in_one_bit to calculate the baseline of the data 
        given in column. See that function for a more detailed description.

        Parameters
        ----------
       column : str, optional
           A string describing what column to apply the funciton on. The 
           default is ''.
        drop_quantile : float, optional
            Quantile under which to drop the minima. Should be between 0 and 1.
            0.5 means drop everything over the median. The default is 0.5.

        Returns
        -------
        float
            Value of the baseline.
        """
        
        col = self._get_column(column)        
        *_, minima = self.calc_baseline_in_one_bit(col, drop_quantile)
                    
        return minima.mean()
    
    @characteristics 
    def calc_multi_baseline(self, column='', drop_quantile=0.5, bits=None, length=None):
        """
        Calculate the local baseline value for each channel. "local" is here 
        defined by cutting the data into multiple pices and calculating it for 
        each pice individually. The pices do not overlap. The user can either 
        decide how many pices to cut the data into using 'bits', or how long 
        should each pice be (in seconds) using 'length'.

        Parameters
        ----------
        channels : str, optional
            A string describing what channel to apply the funciton on. The 
            default is ''.
        drop_quantile : float, optional
            Quantile under which to drop the minima. Should be between 0 and 1.
            0.5 means drop everything over the median. The default is 0.5.
        bits : int, optional
            Ammount of bits into which to chop the data. Calculate the baseline
            on each bit. The default is None.
        length : float, optional
            Length (in seconds) of each bit into which the data gets chopped.
            Calculat the baseline on each bit. The default is None.

        Returns
        -------
        times : array
            times at which the baselines where calculated.
        baselines : array
            baselines.
        
        """
        
        col = self._get_column(column)
        sampling_rate = self._df.metadata.sampling_rate
        N = len(col)
        
        step_length = self._validate_bits_and_length(bits, length, N, sampling_rate)
        
        # calculate the baselines
        baselines = []
        times = []
        for n in range(0, N, step_length):
            theslice = slice(n, min(n+step_length, N))
            *_, minima = self.calc_baseline_in_one_bit(col.values[theslice], drop_quantile)
            baselines.append(minima.mean())
        
            time = (theslice.start + theslice.stop)/2 / sampling_rate
            times.append(time)
                    
        times = np.asarray(times)
        baselines = np.asarray(baselines)
        
        return times, baselines

    @staticmethod
    def calc_baseline_in_one_bit(data, drop_quantile=0.5):
        """
        Finds the local minima of the data in ch and filters out all the ones
        that are over the given drop_quantile. It returns all the local minima 
        and the index at which they happen, and the local minima under the 
        requested quartile, as well as the indexes at which they happen.

        Parameters
        ----------
        data : array
            Data over which to find the baselines.
        drop_quantile : float, optional
            Quantile under which to drop the minima. Should be between 0 and 1.
            0.5 means drop everything over the median. The default is 0.5.

        Returns
        -------
        min_inx : array of ints
            indexes at which the minima happen.
        minima : array
            values of all the local minima.
        filtered_min_inx : array of ints
            indexes at which minima that are under the requested quantile happen.
        filtered_minima : array
            values of the local minima under the given quantile.
        """
        
        min_inx, _ = signal.find_peaks(-data)
        minima = data[min_inx]
                
        # find the requested quantile of the minima
        minima = data[min_inx]
        minima_quantile = np.quantile(minima, drop_quantile)
        
        # keep only the minima under the requested quantile
        filtered_min_inx = min_inx[minima<=minima_quantile]
        filtered_minima = data[filtered_min_inx]
        
        # return filtered_minima.mean()
        
        return min_inx, minima, filtered_min_inx, filtered_minima        
        
    @event_finding
    def find_peaks(self, prominence=5, period_percent=0.6, column=''):
        """
        Finds the peaks in the sata saved in column. See my_find_peaks for more
        info.

        Parameters
        ----------
        prominence, distance : float
            See my_find_peaks for more info.        
        column : str, optional
            A string describing what column to apply the funciton on. The 
            default is ''.

        Returns
        -------
        None.

        """
        
        action_name = 'findpeaks'
        col = self._get_column(column)
        
        # first channel data
        peak_indexes = self.my_find_peaks(self._df.times, col, prominence, period_percent)
                
        self.peaks = peak_indexes
        self._add_process_entry(action_name, prominence=prominence, period_percent=period_percent, column=column)
        
    
    @staticmethod
    def my_find_peaks(times, ch, prominence=5, period_percent=0.6):
        """
        Find the peaks in the signal given by (times, ch). The signal is assumed
        to be fairly clean (a gaussian fitler with sigma=100ms seems to be good
        enough). To do so it does three findpeaks passes:
            1. Find (all) peaks that are above 0mv (assumes a detrended signal)
            # 2. Find peaks that fall above a given threshold. The threshold is 
            # calculated from the data of the previous pass using an otsu 
            # thresholding method to discriminate high peaks and spurious peaks.
            # The otsu threshold will only be used if it falls between two maxima
            # of the distribution of peaks, since it tends to give useless values
            # when the distribution has only one maximum.
            3. Find peaks that lie at least some distance away from the previous
            peak. The distance is calculated as period_percent% of the mode of 
            the period duration, as given by the previous pass.

        NOTE: step 2 is currently disabled.

        Parameters
        ----------
        times : array
            time vector.
        ch : aray
            data vector.
        prominence : float, optional
            Minimum prominence of the peaks in the third pass. Use None to 
            disable prominence check. See scipy.signal.find_peaks for more 
            information on prominence. Default is 5.
        distance : float, optional
            Required minimum distance between peaks in the third step as a 
            fraction of the average distance between peaks found in the first 
            two steps. It's used to avoid finding peaks that are too close 
            together. Use None to disable distance check. See s
            cipy.signal.find_peaks for more information on distance. Default is
            0.6. 

        Returns
        -------
        p_inx : array
            Indexes where the peaks happen.

        """
        
        ## first pass, with threshold 0mv
        p_inx, _ = signal.find_peaks(ch, height=0)
        # peaks = ch.values[p_inx]
        
        ## second pass, with threshold given by otsu
        # threshold = Processors.get_threshold(peaks)
        # p_inx, _ = signal.find_peaks(ch, height=threshold)
        # peaks = ch.values[p_inx]
        t_peaks = times.values[p_inx]

        ## third pass, with minimum distance between peaks
        counts, bins = np.histogram(np.diff(t_peaks))
        bin_centers = bins[:-1] + np.diff(bins) / 2
        period_mode = bin_centers[ np.argmax(counts) ]
        distance_points = int(period_mode * period_percent / (times[1] - times[0])) if period_percent is not None else None
        # p_inx, _ = signal.find_peaks(ch, height=threshold, distance=distance_points)
        
        p_inx, _ = signal.find_peaks(ch, distance=distance_points, prominence=prominence)
                
        return p_inx
    
    # @staticmethod
    # def get_threshold(peaks, fallback_threshold=4):
    #     """Use the otsu method to calculate the threshold for peaks. If the 
    #     value is too big (more than twice the fallback threshold), don't accept
    #     it. If the distribution of peaks has only one maximum, don't accept it.
    #     If the distribution has two or more maxima (including the first point) 
    #     and the threshold doesn't fall between those maxima, don't accept it.
    #     """
    #     threshold = filters.threshold_otsu(peaks)
        
    #     if threshold > 2 * fallback_threshold:
    #         return 2 * fallback_threshold
        
    #     # we will only accept otsu's threshold if we cna detect two peaks in the 
    #     # distribution and the threshold falls between them
    #     counts, bins = np.histogram(peaks, bins='auto')
    #     bin_centers = bins[:-1] + np.diff(bins) / 2
    #     maxima, _ = signal.find_peaks(counts)
    #     # make sure the other peak is not the first point of the distribution
    #     if all(counts[0]>c for c in counts[1:3]):
    #         maxima = np.array( (0, *maxima) )

    #     # if only one maximum was detected, we fallback
    #     if len(maxima) < 2:
    #         return fallback_threshold
        
    #     # if too many maxima were found, keep the largest two
    #     if len(maxima) > 2:    
    #         maxima = sort_by(maxima, counts[maxima])[-2:]
    #         maxima.sort()    

    #     # if at least two maxima were found, accept threshold only if it lies between them
    #     if not( bin_centers[maxima[0]] < threshold < bin_centers[maxima[1]]):
    #         return fallback_threshold
        
    #     return threshold
    
    @property
    def peaks(self):
        """ Peak indexes"""
        return self._peaks
    @peaks.setter
    def peaks(self, peak_indexes):
        if self.peaks is not None:
            print("You've already calculated peaks before. Overriding the previous one.")
        self._peaks = peak_indexes
   
    def get_peak_pos(self):
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        return self._df.times.values[self.peaks]
   
    def get_peak_values(self, column=None):
        """ Returns the value of the series given in column at the peaks that 
        have been calculated. Note you can query the value of a channel for 
        which the peaks have not been calculated, in which case the function 
        will warn and continue working."""
        
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        
        if column is not None and not self._check_if_action_was_performed_on_column('findepakes', column):
            print('WARNING: findpeaks was not performed on this channel')
        
        if column is None: # defalt to calculating peaks on the channel where findpeaks was done
            column = self.get_step_info('findpeaks')['column']
        col = self._get_column(column)
        return col.values[self.peaks]
    
    @characteristics 
    def get_avg_period(self):
        """ Calculate the average (mean) period from a find peaks operation."""
        return np.mean( np.diff(self.get_peak_pos()))
        
    def get_periods(self):
        """ Calculate the cycle duration at each oscillation from a find peaks
        operation."""
        peak_pos = self.get_peak_pos()
        periods = np.diff(peak_pos)
        period_times = peak_pos[:-1] + np.diff(peak_pos ) / 2
        
        return period_times, periods
        
    def get_instant_period(self, outlier_mode_proportion=1.8):
        """ Calculate a trend for the value of the cycle durations of the 
        recording as time passes. Return a linear fit to all the period values
        calculated"""
    
        # find the mode of the periods        
        period_times, periods = self.get_periods()
        period_mode = calc_mode(periods)    
    
        #use the mode to select period values that are not too far from it    
        valid_period_inxs = periods < outlier_mode_proportion*period_mode
        valid_periods = periods[valid_period_inxs]
        
        #calculate the trend and return that as an instant period value
        P = np.polynomial.Polynomial
        trend_poly = P.fit(period_times[valid_period_inxs], valid_periods, 1)
        
        return trend_poly(self._df.times) 
    
    @event_finding
    def get_crossings(self, edge, threshold=5, peak_min_distance=0.5, none_handling='warn', column=None):
        """
        Calculate the point at which the signal crosses the threshold at each
        rising edge or each falling edge, depending on the value of 'edge'. You
        need to have made a findpeaks operation on the data. If the requested 
        column is not the same find_peaks was performed on, the function will 
        warn but continue to work.

        Parameters
        ----------
        inx : int
            1 or 2, the channel for which periods are being calculated.
        edge : str
            either 'falling' or 'rising'. The kind of edge for which the period
            is calculated. If kind='rising', the function will look for a rising
            edge crossing before the peak. If kind='falling', the crossing will
            happen after the peak.
        threshold : float, optional
            Value of the threshold at which the crossing is calculated. 
            Additionally, if a peak value falls under the threshold, that peak
            is ignored. The default is 5, which assumes the data has been 
            detrended.
        peak_min_distance : floar, optional
            The minimum distance between peaks, in units of average distance
            between peaks. After one peak is used, any subsequent peaks closer
            than this value will be ignored. The default is 0.5.
        none_handling : str, optional
            One of 'error', 'warn', 'ignore'. Decides how to handle when no 
            peaks were found. Default is 'warn'.
        column : str or None, optional
            A string describing what column to apply the funciton on. If None 
            the function defaults to the column find_peaks was performed on. 
            The default is None.

        Returns
        -------
        crossings : numpy array
            Array containing indexes at which the requested crossings happen
        """
        
        
        assert edge in ('rising', 'falling'), 'edge must be one of "rising" or "falling"'
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        assert none_handling in ('error', 'warn', 'ignore')
        
        # check if findpeaks was done on the requested channel (for example, gauss_filt)
        # if not, warn and continue
        if column is not None and not self._check_if_action_was_performed_on_column('findepakes', column):
            print('WARNING: findpeaks was not performed on this column')
        
        # default to using the channel on which findpeaks was performed
        if column is None: 
            column = self.get_step_info('findpeaks')['column']
        col = self._get_column(column)
        data = col.values
        
        # retrieve peak data
        peaks = self.peaks
        default_mean_period = -np.inf if edge == 'rising' else np.inf
        mean_period_in_points = round(np.mean(np.diff(peaks))) if peaks.size > 1 else default_mean_period
        
        crossings = []
        prev_peak = -np.inf # to prime the first peak
        
        # iterate over all peaks
        for peak in peaks:
            
            # skip maxima that are too low
            if data[peak] < threshold:
                continue
             
             # skip maxima that are too close together
            if peak - prev_peak < mean_period_in_points * peak_min_distance:
                continue
            
            if edge == 'rising':
                # find rising edge point (before peak)
                interval = data[:peak]
                try:
                    cross = np.nonzero(interval < threshold)[0][-1]
                except IndexError: 
                # this raises when we have the first peak and don't cross the threshold before the start of the signal
                    cross = 0
                                   
            else:
                # find falling edge point (after peak)    
                starting_point = min(peak + int(mean_period_in_points * peak_min_distance), len(data))
                interval = data[:starting_point]
                cross = np.nonzero(interval > threshold)[0][-1]
            
            crossings.append(cross)
            prev_peak = peak
        
        if not crossings: # no peaks were found over the threshold
            msg = f"It's likely no peaks were found over the given threshold value of {threshold}. Or maybe something else is wrong. Are you sure you have peaks in '{self._df.metadata.file.stem}'?"
            
            if none_handling == 'warn':
                cprint(f'&ly {msg}')
            elif none_handling == 'error':
                raise ValueError(msg)
            elif none_handling == 'ignore':
                pass

        crossings = np.asarray(crossings)
        return crossings
    
    @event_finding
    def get_multi_crossings(self, edge, threshold=5, threshold_var=3, 
                            peak_min_distance=0.5, column=None):
        """
        Calculate the point at which the signal crosses the threshold at each
        rising edge or each falling edge, depending on the value of 'edge'. 
        Perform this 23 times in total, distributed simetrically in the interval
        [threshold-threshold_var, threshold+threshold_var]. You need to have 
        made a findpeaks operation on the data. If the requested column is not 
        the same find_peaks was performed on, the function will warn but 
        continue to work.

        Parameters
        ----------
        edge : str
            either 'falling' or 'rising'. The kind of edge for which the period
            is calculated. If kind='rising', the function will look for a rising
            edge crossing before the peak. If kind='falling', the crossing will
            happen after the peak.
        threshold : float, optional
            Value of the threshold at which the crossing is calculated. 
            Additionally, if a peak value falls under the threshold, that peak
            is ignored. The default is 5, which assumes the data has been 
            detrended.
        threshold_var : floar, optional
            Value by which the threshold will be moved upwards and downwards to
            find further corssings. The default is 3.
        peak_min_distance : floar, optional
            The minimum distance between peaks, in units of average distance
            between peaks. After one peak is used, any subsequent peaks closer
            than this value will be ignored. The default is 0.5.
        column : str or None, optional
            A string describing what column to apply the funciton on. If None 
            the function defaults to the column find_peaks was performed on. 
            The default is None.

        Returns
        -------
        crossings : numpy array
            Array containing indexes at which the requested crossings happen
        """
        
        
        assert edge in ('rising', 'falling'), 'edge must be one of "rising" or "falling"'
        assert 'findpeaks' in self.steps, 'You must find peaks first'
        
        # check if findpeaks was done on the requested channel (for example, gauss_filt)
        # if not, warn and continue
        if column is not None and not self._check_if_action_was_performed_on_column('findepakes', column):
            print('WARNING: findpeaks was not performed on this column')
        
        # default to using the channel on which findpeaks was performed
        if column is None: 
            column = self.get_step_info('findpeaks')['column']
        col = self._get_column(column)
        data = col.values
        
        # retrieve peak data
        peaks = self.peaks
        mean_period_in_points = round(np.mean(np.diff(peaks)))
        
        crossings = []
        other_crossings = []
        prev_peak = -np.inf # to prime the first peak
        multiple_thresholds = np.linspace(0, threshold_var, 11)[1:]
        # iterate over all peaks
        for peak in peaks:
            
            # skip maxima that are too low
            if data[peak] < threshold:
                continue
             
             # skip maxima that are too close together
            if peak - prev_peak < mean_period_in_points * peak_min_distance:
                continue
            
            this_other_crossings = []
            if edge == 'rising':
                # find rising edge point (before peak)
                interval = data[:peak]
                try:
                    cross = np.nonzero(interval < threshold)[0][-1]
                        
                except IndexError: 
                # this raises when we have the first peak and don't cross the threshold before the start of the signal
                    cross = 0

                # find corossings around this crossing                
                if cross == 0:
                    # if we are handling the first one, skip searching
                    other_crossings.append(np.full((multiple_thresholds.size*2+1,), np.nan))
                else:
    
                    # find individual crossings
                    # first find crossings below
                    start = max(cross - int(mean_period_in_points * peak_min_distance), 0)
                    short_interval = data[start:cross]
                    for th in multiple_thresholds[::-1]:
                        th = threshold - th
                        
                        other_cross_array = np.nonzero(short_interval < th)[0]
                        if other_cross_array.size == 0:
                            other_cross = np.nan
                        else:
                            # last point bellow threshold
                            other_cross = other_cross_array[-1]
                            other_cross += start #redefine the 0th inxdex
                                                    
                        this_other_crossings.append(other_cross)
                    
                    # append the crossing at the middle
                    this_other_crossings.append(cross)
                    
                    # and now crossings above
                    short_interval = data[cross:peak]
                    for th in multiple_thresholds:    
                        th += threshold
                        
                        other_cross_array = np.nonzero(short_interval > th)[0]
                        if other_cross_array.size == 0:
                            other_cross = np.nan
                        else:
                            # first point above threshold
                            other_cross = other_cross_array[0]
                            other_cross += cross #redefine the 0th inxdex
                        
                        this_other_crossings.append(other_cross)
                    
                    other_crossings.append(np.asarray(this_other_crossings))
                        
            else:
                raise NotImplementedError('There is no multi crossing detection for falling edge yet')
                # find falling edge point (after peak)    
                starting_point = min(peak + int(mean_period_in_points * peak_min_distance), len(data))
                interval = data[:starting_point]
                cross = np.nonzero(interval > threshold)[0][-1]
            
            crossings.append(cross)
            prev_peak = peak
        
        if not crossings: # no peaks were found over the threshold
            raise ValueError(f"It's likely no peaks were found over the given threshold value of {threshold}. Or maybe something else is wrong. Are you sure you have peaks?")

        crossings = np.asarray(crossings)
        other_crossings = np.asarray(other_crossings)
        return crossings, other_crossings
    
    def get_edge_periods(self, edge, threshold=5, peak_min_distance=0.5, column=None):
        """
        Calculate the period as the distance between points at which consecutive
        cycles cross the threshold. The crossing must be either a rising or 
        falling edge, depending on "kind". To perform this action, the user must
        already have performed a findpeaks actions. The crossings are calculated
        with respect to peaks of the signal. Peaks that are under the threshold
        or are too close together are ignored.

        Parameters
        ----------
        inx : int
            1 or 2, the channel for which periods are being calculated.
        edge : str
            either 'falling' or 'rising'. The kind of edge for which the period
            is calculated. If kind='rising', the function will look for a rising
            edge crossing before the peak. If kind='falling', the crossing will
            happen after the peak.
        threshold : float, optional
            Value of the threshold at which the crossing is calculated. 
            Additionally, if a peak value falls under the threshold, that peak
            is ignored. The default is 5, which assumes the data has been 
            detrended.
        peak_min_distance : float, optional
            The minimum distance between peaks, in units of average distance
            between peaks. After one peak is used, any subsequent peaks closer
            than this value will be ignored. The default is 0.5.
        column : str or None, optional
            A string describing what column to apply the funciton on. If None 
            the function defaults to the column find_peaks was performed on. 
            The default is None.


        Returns
        -------
        period_times, periods : tuple
            Tuple containg two arrays: time at which the period is calculated (at
            which the crossing happens) and value of the interval between two 
            consecutive crossings.
        
        """
        data = self._df
        crossings = self.get_crossings(edge, threshold, peak_min_distance, column)
        
        crossing_times = data.times.values[crossings]
        period_times = crossing_times[:-1] + np.diff(crossing_times ) / 2
        periods = np.diff(crossing_times)
        
        return period_times, periods 


    def get_multi_edge_periods(self, edge, threshold=5, threshold_var=3, peak_min_distance=0.5, column=None):
        """
        Calculate the period as the distance between points at which consecutive
        cycles cross the threshold. The crossing must be either a rising or 
        falling edge, depending on "kind". To perform this action, the user must
        already have performed a findpeaks actions. The crossings are calculated
        with respect to peaks of the signal. Peaks that are under the threshold
        or are too close together are ignored.

        Parameters
        ----------
        inx : int
            1 or 2, the channel for which periods are being calculated.
        edge : str
            either 'falling' or 'rising'. The kind of edge for which the period
            is calculated. If kind='rising', the function will look for a rising
            edge crossing before the peak. If kind='falling', the crossing will
            happen after the peak.
        threshold : float, optional
            Value of the threshold at which the crossing is calculated. 
            Additionally, if a peak value falls under the threshold, that peak
            is ignored. The default is 5, which assumes the data has been 
            detrended.
        peak_min_distance : float, optional
            The minimum distance between peaks, in units of average distance
            between peaks. After one peak is used, any subsequent peaks closer
            than this value will be ignored. The default is 0.5.
        column : str or None, optional
            A string describing what column to apply the funciton on. If None 
            the function defaults to the column find_peaks was performed on. 
            The default is None.

        Returns
        -------
        period_times, periods : tuple
            Tuple containg two arrays: time at which the period is calculated (at
            which the crossing happens) and value of the interval between two 
            consecutive crossings.
        
        """
        data = self._df
        crossings, multi_crossings = self.get_multi_crossings(edge, threshold, threshold_var, peak_min_distance, column)
        
        # handle the nans like so:
        temp = data.times.values.copy()
        # add an inf to the end of the array
        temp = np.append(temp, np.nan)
        # replace the nans in crossings with -1 to reference the added nan
        multi_crossings[np.isnan(multi_crossings)] = -1
        multi_crossings = multi_crossings.astype(int)
        
        # the time point in the middle of the oscillation
        crossing_times = temp[crossings]
        period_times = crossing_times[:-1] + np.diff(crossing_times ) / 2
        
        # get array of periods
        multi_crossing_times = temp[multi_crossings]
        all_periods = np.diff(multi_crossing_times, axis=0)

        # filter out rows where we have all nans, to avoid empty slices
        all_nans = np.all(np.isnan(all_periods), axis=1)
        all_periods = all_periods[~all_nans]
        period_times = period_times[~all_nans]

        # the average, std and ptp of each period
        periods = np.nanmean(all_periods, axis=1)
        period_err = np.nanstd(all_periods, axis=1)
        period_ptp = (np.nanmax(all_periods, axis=1) - np.nanmin(all_periods, axis=1)) / 2
        
        return period_times, periods, period_err, period_ptp

    
    def _save_processed_data(self, processed, keep_og, column, action):
        """
        Write the (processed) data form 'processed' into the corresponding 
        column, taking care to replace or keep the original data as 
        needed. If the input data is smaller than the original data, it will
        assume the missing data is due to nans during the calculation and append
        the new data where the original had no nans. This will error out if the
        reason for the size mismatch was differente.

        Parameters
        ----------
        processed : data
            The processed data.
        column : str
            What column was targeted.
        keep_og : Bool
            Whether to keep the original column or not.
        action : str
            what type of processing was performed.

        Returns
        -------
        None.

        """
        
        og_column_name = self._get_column_name(column)
        
        # handle the case where the input had nans
        if processed.size != self._df.rec.size:
            og_column = self._get_column(column)
            nan_locs = np.isnan(og_column)
            
            if np.sum(~nan_locs) != processed.size:
                raise ValueError("The size of the input data couldn't be matched to the non nan values in the original data")
            
            full_nans = np.full(og_column.shape, np.nan)
            full_nans[~nan_locs] = processed
            processed = full_nans
            
        # define behaviour regarding column overwritting
        if keep_og:
            new_name = self._ensure_name_is_new(og_column_name + '_' + action, self._df.columns)
            
        else:
            new_name = og_column_name
                
        # write columns
        self._df[new_name] = processed
    
    @staticmethod
    def _ensure_name_is_new(name, existing):
        """ Checks if the suggested name is new. If not, it appends a number at
        the end such that the returned name is unique."""
        new_name = name
        i = 1
    
        while new_name in existing:
            i+=1
            new_name = name+str(i)
        
        return new_name
    
    def _add_process_entry(self, action, **kwargs):
        """
        Add an entry to the processing list detailing what was done and what
        parameter were used.

        Parameters
        ----------
        action : str
            The action that was performed, i.e. the name of the processing step.
        **kwargs : 
            The arguments to the processing step.

        Returns
        -------
        None.
        
        """
        
        step = {'step':action, **kwargs}
        self.info = step
        
    def _get_column_name(self, column):
        """
        Return the full name of the column that matches column.

        Parameters
        ----------
        column : str
            A string describing the column. For example, 'detrend' will return
            'rec_detrend'. An empty string will return the unprocessed column.

        Returns
        -------
        Column name.

        """
            
        if not isinstance(column, str):
            raise TypeError('column must be a string')
        
        # get a list of the colums that hold data, minus the leading 'rec_'
        column_options = set(re.sub(r'rec_?', '', x) for x in self._df.columns if x not in self._non_data_columns)
        column = column.strip('_') # strip leading '_' in case there were any
        if column not in column_options:
            print(f'{column} is not an available column. Choose one of {column_options}. Returning default raw channels.')
            column = ''
        
        column_name = 'rec_'+column if column else 'rec'
        
        return column_name
        
    def _get_column(self, column):
        """
        Gets the series corresponding to the column defined by column. Uses
        _get_column_name to figure out what those channels are.

        Parameters
        ----------
        channels : str
            see _get_column_name.

        Returns
        -------
        pandas.Series
        """
        column_name = self._get_column_name(column)
        return self._df[column_name]

    def get_step_info(self, step, last=True):
        """
        Returns the dictionary that stores the arguments corresponding to the 
        queried processing step. If multiple steps with the same name were
        registered, it returns the last one if last=True, or all of them 
        otherwise. 

        Parameters
        ----------
        step : str
            step name.
        last : Bool, optional
            Whether to return only the last step with the corresponding name,
            or all the matching ones. If last=True, return type is a dict, if
            last=False, return type is a tuple of dicts. The default is True.

        Returns
        -------
        step_info: dict, tuple of dicts

        """
        if step not in self.steps:
            raise ValueError(f'The step "{step}" has not been performed yet.')
            
        matches = (len(self.steps) - 1 - i for i, x in enumerate(reversed(self.steps)) if x==step)
        if last:
            return self.info[next(matches)]
        else:
            return tuple( reversed( [self.info[inx] for inx in matches] ) )

    def _check_if_action_was_performed_on_column(self, action: str, column: str) -> bool:
        """
        Checks if an action was performed on a given column and returns a 
        boolean result. The caller has to decide what to do with this 
        information.
        """
    
        if action not in self.steps:
            return False
        
        #check if the action found was done on the correct column
        steps = self.get_step_info(action, last=False)
        if all(column!=step['column'] for step in steps):
            return False # I know I can just return the result of the all call, but this is more explicit
        else:
            return True

    @staticmethod
    def _validate_bits_and_length(bits, length, N, sampling_rate):
        """
        Intended to use in functions where the data is going to be chopped up 
        into pices and the function can decide if the pices are defined by a
        total count or by their duration. Both can't be defined at the same time

        Parameters
        ----------
        bits : int
            How many pices to chop the data into.
        length : float
            How long should each pice of data be (in seconds).
        N : int
            Size of the data to chop into pices.

        Returns
        -------
        step_length : int
            how many points long is the step needed to conform with either bits
            of length.            
        """
        
        if bits is None and length is None:
            raise ValueError('You must give bits or length')
        if bits is not None and length is not None:
            raise ValueError("You can't define both bits and length. Choose one.")
        
        # calculate the length of the step 
        if length is not None:
            step_length = int(length * sampling_rate)
        if bits is not None:
            assert isinstance(bits, numbers.Integral)
            step_length = N // bits
            
        return step_length
    
    def plot(self, processing_steps=None, ax=None):
        """
        Plot the data. If current was injected, plot that too. Plot the processing
        steps requested, or just the raw data.

        Parameters
        ----------
        processing_steps : str, iterable or None, optional
            Define what processing step(s) to plot. If None, just plot the raw 
            data. If 'all', plot raw and all processing steps. If an iterable, 
            it should contain the list of the steps as defined in steps. Use 
            'raw' to plot the raw data. The default is None.
        ax : matplotlib.Axes or None, optional
            Provide an axis to plot into. If none is given, create a figure and
            an axis. Default is None.

        Returns
        -------
        axis object handle.

        """
        
        if ax is None:
            fig, ax = plt.subplots()
        
        data = self._df
        metadata = self._df.metadata
        
        # process input
        if processing_steps is None:
            processing_steps = ['raw']
        elif processing_steps == 'all':
            processing_steps = [c for c in self._df.columns if c not in self._non_data_columns]
        elif isinstance(processing_steps, str):
            processing_steps = [processing_steps]
        
        # plot processing steps
        for step in processing_steps:
            if step == 'raw':
                step = 'rec'
            ax.plot(data.times, data[step], label=step)
         
        # plot current injection
        for i, perturbation_col in enumerate(('pert', 'dig0', 'dig1')):
            
            if not all(data[perturbation_col]==0):
                ax.plot(data.times, data[perturbation_col], 'k', label=metadata._labels[2+i])
        
        # format axes
        ax.set_xlabel(metadata._labels[0])
        ax.set_ylabel(metadata._labels[1])
        ax.set_title(metadata.file.stem)
        ax.legend()
        
        ax.set_xlim(data.times.values[0], data.times.values[-1])
        
        return ax    
    
    #####################
    ### REMOVED STUFF ###
    #####################
    
    ### Processing things
    # average_detrend
    # varying_size_rolling_average
    
    ### Dual channel things
    # cross_correlation
    # multi_cross_correlation_lag
    # calc_corr_lag
    # calc_phase_difference

#%% Load functions

def load_data(file, sweep=None, channel=0):
    """
    Load data from an abf file. Builds the data into a dataframe with columns
    'time', 'rec' and 'pert' containng the time vector, the actual recording
    and the perturbation (usually imput current), respectively. The dataframe
    is overloaded with two accessors, one for processing and one for recording 
    metadata. The latter gets automatically populated with data from the 
    recording file and can be further enriched with an 'info.csv' file, where a
    column labeled 'recoding' contains the recording name and the following 
    columns contain the relevant metadata.

    Parameters
    ----------
    file : str or Path-like
        Path to the file to be loaded.
    sweep : int or None, optional
        What sweep to load. No check is done to assure the requested sweep 
        exists, so pyabf will error out if it does not. If sweep is None, all
        sweeps will be loaded and concatenated. The default is None.
    channel : TYPE, optional
        What channel to load. No check is done to assure the requested channel
        exists, so pyabf will error out if it does not. The default is 0.

    Returns
    -------
    data : pandas.DataFrame
        The loaded data

    """
    # Load data
    abf = pyabf.ABF(file)
    info = extract_info_line(file)
    
    # Extract recording
    times, rec, pert, dig0, dig1 = extract_data(abf, sweep, channel)
    interval = extract_target_interval(info, abf)
     
    # Pack everything into a DataFrame
    data = pd.DataFrame(data={'times':times, 'rec':rec, 'pert':pert
                             'dig0':dig0, 'dig1':dig1})
    data = data[interval].reset_index(drop=True)
    data.times -= data.times.values[0]
    
    # Fill the DataFrame with info
    extract_recording_info(data, abf)
    
    data.metadata.info = info
    data.metadata.file = file
    data.metadata.channel = channel
    data.metadata.sweep = sweep
    
    return data

def extract_data(abf, sweep, channel):
    """ Extract data from the recording. If multiple sweeps are available,
    concatenate them all. If a sweep was specified, exctract only that one, 
    provided it exists. 
    Extract data only from specified channel.
    """
    
    if sweep is not None and sweep not in abf.sweepList:
        raise ValueError(f'Invalid sweep number provided: {sweep}. Choose one of {abf.sweepList}.')
    
    # extract requested sweep, if requested
    if sweep is not None:
        abf.setSweep(sweep, channel)
        
        times = abf.sweepX
        rec = abf.sweepY
        pert = abf.sweepC
        dig0 = abf.sweepD(0)
        dig1 = abf.sweepD(1)
        
    # else, concatenate all sweeps
    else:
        times = []
        rec = []
        pert = []
        dig0 = []
        dig1 = []
        
        for sweep in abf.sweepList:
            # Set sweep and channel, extract tmes and data
            # We could use getAllXs and getAllYz, but there's no equivalent for C
            abf.setSweep(sweep, channel, absoluteTime=True)
            
            times.append(abf.sweepX)#
            rec.append(abf.sweepY)
            pert.append(abf.sweepC)
        
        times = np.concatenate(times)
        rec = np.concatenate(rec)
        pert = np.concatenate(pert)
        dig0 = np.concatenate(dig0)
        dig1 = np.concatenate(dig1)
        
    return times, rec, pert, dig0, dig1

def extract_target_interval(info, abf):
    """ Extract the interval of the data to use, if we need to cut it. The 
    interval should be specified inthe column 'range_min'."""
    
    # extract interval
    if info is not None and 'range_min' in info:
        interval = info.range_min
    else:
        interval = None
    
    # Cut data at required points
    if interval is not None and isinstance(interval, str):
        start_min, end_min = map(float, interval.split('-'))
        start = int(start_min * 60 * abf.sampleRate)
        end = int(end_min * 60 * abf.sampleRate)
    else:
        start = None
        end = None
    
    return slice(start, end)

def extract_recording_info(data, abf):
    """ Extracts a number of parameters from the abf file and stores them in 
    the .metadata  attribute.
    Add date of recording, sampling rate (and clone it into raw sampling rate), 
    and duration.   
    """
        
    data.metadata.sweep_count = abf.sweepCount
    data.metadata.sweep_duration_sec = abf.sweepLengthSec
    
    data.metadata.channel_count = abf.channelCount
    
    data.metadata.rec_datetime = abf.abfDateTime
    data.metadata.raw_sampling_rate = abf.sampleRate
    data.metadata.protocol = abf.protocol
    data.metadata.sampling_rate = data.metadata.raw_sampling_rate
    data.metadata.duration = data.times.values[-1]
    
    data.metadata._labels = [abf.sweepLabelX, abf.sweepLabelY, abf.sweepLabelC]

def extract_info_line(file):
    """ Check if an info file is available and open it up. Look for the line 
    with the infor for the current file and return it."""
    
    if not isinstance(file, Path):
        file = Path(file)
        
    # try to find an info file
    info_file = file.parent / 'info'
    extensions = 'xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt', 'csv'
    
    for ext in extensions:
        if info_file.with_suffix('.' + ext).exists():
            info_file = info_file.with_suffix('.' + ext)
            break
    else:
        return None
    
    # load the file:
    if ext == '.csv':
        info = pd.read_csv(info_file)
    else:
        info = pd.read_excel(info_file)
    
    # force recoding names to string type
    info.recording = info.recording.astype(str)
    info = info.set_index('recording')
    
    # extract info line
    try:
        info_line = info.loc[file.stem]
    except KeyError:
        cprint(f'&ly Warning: we found an info file but the recording {file.stem} has no entry.')
        return None            
    
    return info_line
