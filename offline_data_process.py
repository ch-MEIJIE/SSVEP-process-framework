# %% [markdown]
# # SSVEP Decoding Framework
# 
# I want to build a general pipeline for SSVEP EEG signal decoding, in which it is easy to complete basic EEG processing stages like:
# 
# * cutting and slicing your data
# * filtering data
# * applying feature extraction method
# * matching pattern and get result
# 
# The key inspiration of the framework design ethic is modular, and I want to take the manually parts like experimental information, special filters and feature extraction methods out of the main executing. So that it's much easier to re-write or add new minds into the framework. In the other words, it is a framework that flexible, easy for new learner and try something new.
# 
# There are 3 main class defined in this framework. If you just focus on get model for online experiment or do some cross validation, the `data_runner` class and `data_cross_validation` class are what you need, you can just read through and run them. In the `filter_apply` class, you can configure your own time-filter parameters, and what I must admit is that the filter parameters in the current version is not the best, and haven't been optimized at all! 
# 
# You may notice that there are several functions above the main classes, they can be named as helper functions. I can move them together and separate them to a helper collection and make the main program clear.

# %% [markdown]
# ## Import necessary external packages here
# 
# In this framework, I try to use external packages as little as possible. Compared to use the toolbox like mne, it's a bit of troublesome, but not too much. Jump out of the mne processing and data framework can make you understand the data route more clearly.
# 
# However, if you like, you can feel free add some packages for boosting the function of the framework.

# %%
from multiprocessing import set_forkserver_preload
from os.path import join as pjoin
import numpy as np
import scipy.signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.io import loadmat
import json
from collections import defaultdict
from matplotlib import pyplot as plt

# Manully packages and funtions
from helper_functions import *


# %%
class result_analyser():
    """ 
    A helper class for analyzing the classification results.
    """
    def __init__(self,labels,CV_loops,epoch_num,**kwargs): 
        """ Initial a result_analyser object

        Args:
            labels (array like vector): Triggers, which should be recorded from EEG amplifier. The triggers is a arrary which associating with how you perform your experiment.
            CV_loops (int): Numbers of CV (Cross validation) loops
            epoch_num (int): Numbers of epochs contained in single trial
        """
        self.labels = list(labels)
        self.epoch_num = epoch_num
        self.CV_loops = CV_loops
        self.acc_storage = np.zeros(self.CV_loops)
        self.trial_counter = 0
        
        if "LDA" in kwargs:
            self.LDA_model_builder = LDA_trainer()
        if "OVERLAP" in kwargs:
            self.overlap_buffer = np.zeros((len(self.labels),kwargs["OVERLAP"]))
            self.overlap_size = kwargs["OVERLAP"]
        else:
            self.overlap_buffer = np.zeros((len(self.labels),1))
            self.overlap_size = 1
        
        self.result_matrix = np.zeros((CV_loops,len(self.labels),self.epoch_num-self.overlap_size+1))
    
    def overlop_trials(self,vector,epoch_iter):
        if epoch_iter == 0:
            self.overlap_buffer = np.zeros((len(self.labels),self.overlap_size))
        self.overlap_buffer[:,epoch_iter%self.overlap_size]=vector
    
    def result_decide(self,coef_vector,CV_iter,trial_iter,epoch_iter):
        """
        Determine whether the classification results match the triggers.

        Args:
            coef_vector (numpy array): A vector saved the correlation coefficients between the single trial and templates.
            CV_iter (int): Current CV loop index, should be smaller than self.CV_loops.
            trial_iter (int): Current trial index in a loop, should be smaller than length of self.labels.
            epoch_iter ([type]): Current epoch index in a trial, should be smaller than self.epoch_num.
        """

        self.trial_counter += 1
        self.overlop_trials(coef_vector,epoch_iter)
        if epoch_iter >= self.overlap_size-1:
            summed_coef_vector = np.sum(self.overlap_buffer,axis=1)
            _result = self.labels[np.argmax(summed_coef_vector)]
            if _result == trial_iter:
                self.result_matrix[CV_iter, self.labels.index(trial_iter), epoch_iter-self.overlap_size+1] = 1
                label = 1
            else:
                label = 0
            
            if "LDA_model_builder" in dir(self):
                self.LDA_model_builder.train_data_collector(summed_coef_vector,label)

        if self.trial_counter % len(self.labels) == 0 and epoch_iter == self.epoch_num-1:
            self.acc_storage[CV_iter] = np.mean(self.result_matrix[CV_iter,:,:])
            print('ACC of the {} cross validation loop is: {}'.format(CV_iter,self.acc_storage[CV_iter]))
        
    def ACC_calculate(self):
        """
        Callable function to calculate the accuracy among all cross validation loops.
        """
        self.overall_ACC = np.mean(self.acc_storage)
        print('Overall ACC of current data is {}'.format(self.overall_ACC))

# %%
class data_trainer():
    """
    Base class to load, pre-process, extract features and calculate templates from rawdata.
    You can view this class as a base class and inherit it to develop your own developing process
    """
    def __init__(self, usrname, data_path, block_num, spatial_filter_type='TRCA') -> None:
        """Parameter initial function
        
        Note: 
        1. The raw folder structure should be like: ParentFolder/usrname/block'N'/EEG.mat
        2. Only data format '.mat' is supported now. You can use mne package or Matlab to load the raw data and convert it to .mat format.
        
        Args:
            usrname (str): volunteer's name
            data_path (str): parent folder's path
            block_num (str): assoicated with how you named your data subfolder, in the example data folder structure, the parameter here is block'N'
            spatial_filter_type (str, optional): The method to extract data features. Defaults to 'TRCA'.
            cross_validation (bool, optional): whether to perform cross validation. Defaults to True.
        """
        with open('config.json') as file:
            self.data_config = json.load(file)
        data_dir = pjoin(data_path,usrname,block_num,'EEG.mat')
        self.raw_data = loadmat(data_dir)
        self.raw_data = self.raw_data['EEG'][0]
        self.event = self.raw_data['event'][0]
        self.event_size = self.event.shape[0]
        self.data = self.raw_data['data'][0]
        # Read experiment paramter setting in json file
        self.blocks_in_data = self.data_config['blocks_in_data']
        self.epochs_in_data = self.data_config['epochs_in_trials']
        self.slice_data_storage = dict()
        self.template_storage = dict()
        self.sample_rate = self.raw_data['srate'][0][0][0]
        self.epoch_length = int(self.data_config['epoch_length'][block_num]*self.sample_rate)
        self.visual_delay = int(0.14*self.sample_rate)
        # Initail time filter paramters here
        self.filter_object = filter_applyer(self.sample_rate,7.0,80.0,64,filter_type='IIR')
        self.spatial_filter_type = spatial_filter_type
        self.template_storage = dict()

        self.data_slice()
   
    def data_slice(self):
        """
        Method to extract data from continuous dataset. In most of conditions, you should not modify this if you follow the guidance.
        
        Note: 
        
        Data frame structure in matlab is not supported in python, it would be transformed into dict format when loaded by scipy.io.loadmat method.
        """
        for event_iter in range(self.event_size):
            event_type = self.event[event_iter][0][0][0]
            event_time_stamp = int(self.event[event_iter][0][1][0][0])
            epoch_cut = self.data[:,event_time_stamp+self.visual_delay:event_time_stamp+self.visual_delay+self.epoch_length]
            # Zero-mean (Must DO! Especially when use FIR filter.)
            epoch_cut = epoch_cut-np.mean(epoch_cut,axis=-1,keepdims=True)
            filtered_epoch_cut = self.filter_object.filter_apply(epoch_cut)
            # Create a new key-value pairs which value is a empty list to save data
            # If key (i.e. event_type is existed in current dict, this line would be overpass)
            event_list = self.slice_data_storage.setdefault(event_type, list())
            event_list.append(filtered_epoch_cut)
        self.event_series = self.slice_data_storage.keys()
        print('Data sliced ready!')
        print('Total number of events: {}'.format(len(self.slice_data_storage)))
    
    def trainer(self):
        """
        Method to train feature extracting model by iteracting the dict keys, which map to all kind of triggers data.
        """
        self.spatial_filters = dict()
        for train_trial_iter in self.event_series:
            self.spatial_filters[train_trial_iter] = self.feature_extract(self.slice_data_storage[train_trial_iter])
            self.template_calculate(self.slice_data_storage[train_trial_iter], train_trial_iter)

    def feature_extract(self,data):
        """
        Define of call your manully feature extraction methods here.
        """
        if self.spatial_filter_type == 'TRCA':
            return trca_matrix(data)
        if self.spatial_filter_type == 'TDCA':
            pass
        raise Exception('Method not define, you can define it manually!')
    
    def template_calculate(self, train_data, event_type):
        self.template_storage[event_type] = np.mean(train_data, axis=0)
        self.template_events = list(self.template_storage.keys())
    
    def train_result_get(self):
        return self.spatial_filters, self.template_storage    

# %%
class data_cross_validation(data_trainer):
    
    def cross_validation_runner(self):
        self.dataset_split_index = train_test_split(split_type='K_Fold',test_sample_num=self.epochs_in_data, total_sample_num=self.epochs_in_data*self.blocks_in_data)
        self.result_saver = result_analyser(self.event_series,self.dataset_split_index.shape[0],self.epochs_in_data)
        for cross_validation_iter in range(self.dataset_split_index.shape[0]):
            self.CV_iter = cross_validation_iter
            print('cross validation loop: {}'.format(cross_validation_iter))
            validation_index = self.dataset_split_index[cross_validation_iter,:]
            self.trainer(1-validation_index)
            self.tester(validation_index)
        self.result_saver.ACC_calculate()
    
    def trainer(self,select_index):
        self.spatial_filters = dict()
        for train_trial_iter in self.event_series:
            self.spatial_filters[train_trial_iter] = self.feature_extract(np.array(self.slice_data_storage[train_trial_iter])[select_index==1,:,:])
            self.template_calculate(np.array(self.slice_data_storage[train_trial_iter])[select_index==1,:,:],train_trial_iter)
    
    def tester(self,select_index):
        self.corrcoef_storage = dict()
        for test_trial_iter in self.event_series:
            test_epoches = np.array(self.slice_data_storage[test_trial_iter])[select_index==1,:,:]
            corrcoef_list = self.corrcoef_storage.setdefault(test_trial_iter, list())
            for test_epoch_iter in range(test_epoches.shape[0]):
                coef_vector = pattern_match(test_epoches[test_epoch_iter,:,:],self.spatial_filters,self.template_storage)
                corrcoef_list.append(coef_vector)
                self.result_saver.result_decide(coef_vector,self.CV_iter,test_trial_iter,test_epoch_iter)

# %%
# Test cross_validation class
cross_validation_tester = data_cross_validation('mengqiangfan','./ThesisData/','block4')
cross_validation_tester.cross_validation_runner()


# TODO: Add LDA module
# TODO: Test TDCA

class simulated_online(data_cross_validation):
    
    def __init__(self):
        pass
# %%
