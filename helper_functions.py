# %%
from os.path import join as pjoin
from re import S
from scipy.io import loadmat
import numpy as np
import scipy.signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from functools import wraps
import json
import warnings

from wheels.load_data_helper import load_neuracle_data

class LDA_trainer():
    def __init__(self):
        self.train_data = list()
        self.labels = list()
    
    def extract_sample(self, coef_vector):
        max_data = np.max(coef_vector)
        label_predict = np.argmax(coef_vector)
        coef_vector = np.delete(coef_vector,label_predict)
        submax_data = np.max(coef_vector)
        return max_data, submax_data
    
    def train_data_collector(self, coef_vector, label):
        max_data, submax_data = self.extract_sample(coef_vector)
        self.train_data.append([max_data,submax_data])
        self.labels.append(label)
    
    def train_model(self):
        self.linear_model = LinearDiscriminantAnalysis()
        self.linear_model.fit(np.array(self.train_data),np.array(self.labels))
    
    def test_model(self, test_sample):
        max_data, submax_data = self.extract_sample(test_sample)
        predict_label = self.linear_model.predict([[max_data, submax_data]])
        return predict_label[0]

def trca_matrix(X):
    """ TRCA kernel function

    Args:
        X (Numpy array): Three dimsional ndarry matrix, shape as (n_trials, n_channels, n_samples)

    Returns:
        spatial_filter(Numpy array): A ndarray vector, shape as (n_channels, )
    
    REF:
    [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao and T. -P. Jung, "Enhancing Detection of SSVEPs for a High-Speed Brain Speller Using Task-Related Component Analysis," in IEEE Transactions on Biomedical Engineering, vol. 65, no. 1, pp. 104-112, Jan. 2018, doi: 10.1109/TBME.2017.2694818.
    """
    n_chans = X.shape[1]
    n_trial = X.shape[0]
    S = np.zeros((n_chans, n_chans))
    # Computation of correlation matrices
    for trial_i in range(n_trial):
        for trial_j in range(n_trial):
            x_i = X[trial_i, :, :]
            x_j = X[trial_j, :, :]
            S = S + np.dot(x_i, x_j.T)
    X = np.transpose(X, (1, 2, 0))
    X1 = X.reshape((n_chans, -1),order='F')
    X1 = X1 - np.mean(X1, axis=1, keepdims=True)
    Q = np.dot(X1, X1.T)
    S = np.matrix(S)
    Q = np.matrix(Q)
    # TRCA eigenvalue algorithm
    [W, V] = np.linalg.eig(np.dot(Q.I,S))

    spatial_filter = V[:, 0].reshape(-1)

    return spatial_filter

# %%
def train_test_split(split_type='K_Fold',test_sample_num = None, total_sample_num = None):
    """[summary]

    Args:
        split_type (str, optional): Key words of the cross_validation method. Defaults to 'K_Fold'.
        test_sample_num (int, optional): the number of test samples (trials) contained in one cross validatioon seperation. Defaults to None.
        total_sample_num (int, optional): total samples (trials) of your dataset. Defaults to None.

    Raises:
        Exception: The sample cannot be divided into test subsets
        Exception: Split_type was not defined, use 'K_Fold' or define it manually

    Returns:
        numpy array: A matrix shape as (total_sample_num/test_sample_num, total_sample_num)
    """
    if split_type == "K_Fold":
        if total_sample_num%test_sample_num != 0:
            raise Exception('The sample cannot be divided into test subsets')
        else:
            folders_num = int(total_sample_num/test_sample_num)
            split_index = np.zeros((folders_num,total_sample_num))
            for folder_index in range(folders_num):
                split_index[folder_index, folder_index*test_sample_num:(folder_index+1)*test_sample_num] = np.ones((test_sample_num,))
        return split_index
    
    raise Exception('split_type was not defined, use \'K_Fold \' or define it manually')

# %%
# Test function for train_test_split()
split_index = train_test_split(test_sample_num=5,total_sample_num=30)

# %%
class filter_applyer():
    """
    A filter warp class, you can design and build time-filter by modify or inherit and rewrite the _filter_design method. With this class, you can easily try different filter paramters setting without break the structure.

    Attention, this filter_appler only support band-pass filter paramter, for it is the most common type in EEG processing.
    """
    def __init__(self,sample_rate,low_cut_frequency,high_cut_frequency,filter_order,filter_type = 'FIR') -> None:
        """
        Args:
            sample_rate (float): Sample rate of filter
            high_cut_frequency (float): High cut-off frequence of the band-pass filter
            low_cut_frequency (float): Low cut-off frequence of the band-pass filter
            filter_order (int): Filter order of FIR filter
            filter_type (str,option): Choose a filter type, in this veision only support FIR which is also default.
        """
        self.sample_rate = sample_rate
        self.filter_type = filter_type
        self.high_cut_frequency = high_cut_frequency
        self.low_cut_frequency = low_cut_frequency
        self.filter_order = filter_order
        self.filter_b = dict()
        self.filter_a = dict()
        self._filter_design()

            
    def _filter_design(self):
        """
        Private method of this class, it would be automatic called by init function when an new filter is created.
        Also, you can manually Design the filter here. Current version support both FIR and IIR filters design.
        """
        # Nyquist rate of signal
        nyq_rate = self.sample_rate/2.0
        # 1-D array cut of
        # freq_pass = [self.low_cut_frequency/nyq_rate, self.high_cut_frequency/nyq_rate]
        # freq_stop = [freq_pass[0]-2/nyq_rate,freq_pass[1]+6/nyq_rate]
        high_pass = self.low_cut_frequency
        high_cut = self.low_cut_frequency - 4
        low_pass = self.high_cut_frequency
        low_cut = self.high_cut_frequency + 8
        if self.filter_type == 'FIR':
            if self.filter_order%2 == 0:
                self.filter_order+=1
            # Get the fir filter coef
            taps = scipy.signal.firwin(self.filter_order, high_pass, window='hamming', pass_zero='highpass')
            self.filter_b['highpass'] = taps
            self.filter_a['highpass'] = 1
            taps = scipy.signal.firwin(self.filter_order, low_pass, window='hamming', pass_zero='lowpass')
            self.filter_b['lowpass'] = taps
            self.filter_a['lowpass'] = 1
        elif self.filter_type == 'IIR':
            ord, wn = scipy.signal.cheb1ord(high_pass/nyq_rate, high_cut/nyq_rate, 4, 30)
            self.filter_b['highpass'],self.filter_a['highpass'] = scipy.signal.cheby1(ord,0.5,wn,btype='highpass')
            ord, wn = scipy.signal.cheb1ord(low_pass/nyq_rate, low_cut/nyq_rate, 4, 30)
            self.filter_b['lowpass'],self.filter_a['lowpass'] = scipy.signal.cheby1(ord,0.5,wn,btype='lowpass')
            
            
    def filter_apply(self,data):
        """Callable function for filtering data.
        Use this function when you need to filter data after the class object is created.

        Args:
            data (numpy array): The data to be filtered. Shape as n_chan

        Returns:
            numpy array: filterd data, same shape as input data.
        """
        filter_data = scipy.signal.filtfilt(self.filter_b['highpass'],self.filter_a['highpass'],data)
        filter_data = scipy.signal.filtfilt(self.filter_b['lowpass'], self.filter_a['lowpass'], filter_data)
        return filter_data

# %%
def pattern_match(testsample,spatial_filters,template_storage):
    """ 
    Function for pattern matching, input a single trial test sample, then filtering with the pre-trained spatial filter, finally calculated person correlation with all the template signal(using np.corrcoef method)

    Args:
        testsample (Numpy array): A numpy array of single test sample, shape as (N_chan,N_samples)
        spatial_filters (dict): spatial filter dictionary storage, the keys are the trial indexes, values are corresponding spatial filters which trained from feature extracted method 
        template_storage (dict): template dictionary storage, the keys are the trial indexes, values are corrsponding spatial filters which calculated from train samples.

    Raises:
        TypeError: Testsample should be a two dimensional vector

    Returns:
        Numpy array: correlation coefficient storage vector, shape as (N_trials,1) 
    """
    if len(testsample.shape) != 2:
        raise TypeError('Testsample should be a two dimensional vector')
    corrcoef_storage = list()
    spatial_filter = np.squeeze(np.array(list(spatial_filters.values())))
    for template_iter in template_storage.keys():
        corrcoef_storage.append(np.corrcoef(np.dot(spatial_filter,testsample).reshape(1,-1),np.dot(spatial_filter,template_storage[template_iter]).reshape(1,-1))[0,1])
    
    corrcoef_storage = np.array(corrcoef_storage)

    return corrcoef_storage

# %%
class data_preprocessor():
    """
    Base class to load, pre-process, extract features and calculate templates from rawdata.
    You can view this class as a base class and inherit it to develop your own developing process
    """
    def __init__(self, usrname, data_path, block_num, file_name='EEG.mat', time_domain_filter = True, extended_bounds = 0, data_type = 'offline'):
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
        self.usrname = usrname
        self.data_path = data_path
        self.block_num = block_num
        self.file_name = file_name
        
        self.time_domain_filter = time_domain_filter
        self.extended_bounds = extended_bounds
        self.data_type = data_type
        if self.data_type == 'offline':
            self.slice_data_storage = dict()
            self.extended_bounds = extended_bounds
        elif self.data_type == 'simu_online':
            self.slice_data_storage = dict()
            self.slice_data_storage['data'] = list()
            self.slice_data_storage['label'] = list()       
    
    def read_config_file(self, file_name = 'config.json'):
        with open(file_name) as file:
            self.data_config = json.load(file)
        self.blocks_in_data = self.data_config[self.data_type]['blocks_in_data']
        self.epochs_in_data = self.data_config[self.data_type]['epochs_in_trials']
        # epoch duration in unit second(s)
        self.epoch_duration = self.data_config['epoch_length'][self.block_num]
    
    def read_data(self):
        self.data_dir = pjoin(self.data_path,self.usrname, self.block_num, self.file_name)
        EEG_data_package = loadmat(self.data_dir)
        self.EEG_data_package = EEG_data_package['EEG'][0]
        self.event = self.EEG_data_package['event'][0]
        self.evnet_size = self.event.shape[0]
        self.data = self.EEG_data_package['data'][0]
        self.sample_rate = self.EEG_data_package['srate'][0][0][0]
        self.epoch_length = int(self.epoch_duration*self.sample_rate)
        self.visual_delay = int(0.14*self.sample_rate)
        if self.time_domain_filter == True:
            self.initial_filters()
    
    def initial_filters(self):
        self.filter_object = filter_applyer(self.sample_rate, low_cut_frequency=7.0, high_cut_frequency=80.0, filter_order=64, filter_type='IIR')
    
    def slice_data(self,):
        if self.data_type == 'offline':
            for event_iter in range(self.evnet_size):
                event_type = self.event[event_iter][0][0][0]
                event_time_stamp = int(self.event[event_iter][0][1][0][0])
                epoch_cut = self.data[:,event_time_stamp+self.visual_delay:event_time_stamp+self.visual_delay+self.epoch_length+self.extended_bounds]
                epoch_cut = epoch_cut-np.mean(epoch_cut,axis=-1,keepdims=True)
                if self.time_domain_filter == True:
                    epoch_cut = self.filter_object.filter_apply(epoch_cut)
                current_event_cache = self.slice_data_storage.setdefault(event_type, list())
                current_event_cache.append(epoch_cut)
            self.event_series = self.slice_data_storage.keys()
            print('Note: The raw continuous data has been sliced by its event triggers')   
            print('Note: There are {} unique triggers in raw data'.format(len(self.event_series)))
            data_check_cache = []
            for event_iter in self.event_series:
                data_check_cache.append(len(self.slice_data_storage[event_iter]))
            data_check_cache = np.array(data_check_cache)
            unique_lens = np.unique(data_check_cache)
            if unique_lens.shape[0] == 1:
                print('Note: Data check passed, all triggers have same epochs, which is {}'.format(unique_lens[0]))
            else:
                warnings.warn('Imbalanced epochs check! The training later may go wrong, carefully check your data!')
        elif self.data_type == 'simu_online':  
            for event_iter in range(self.evnet_size):
                event_type = self.event[event_iter][0][0][0]
                event_time_stamp = int(self.event[event_iter][0][1][0][0])
                epoch_cut = self.data[:,event_time_stamp+self.visual_delay:event_time_stamp+self.visual_delay+self.epoch_length]
                epoch_cut = epoch_cut-np.mean(epoch_cut,axis=-1,keepdims=True)
                if self.time_domain_filter == True:
                    epoch_cut = self.filter_object.filter_apply(epoch_cut)
                self.slice_data_storage['data'].append(epoch_cut)
                self.slice_data_storage['label'].append(event_type)               
    
    def send_data(self):
        data_info = dict()
        if self.data_type == 'offline':
            data_info['data_type'] = 'offline'
            data_info['epochs_in_data'] = self.epochs_in_data
            data_info['blocks_in_data'] = self.blocks_in_data
        elif self.data_type == 'simu_online':
            data_info['data_type'] = 'offline'
            data_info['epochs_in_data'] = self.evnet_size
            data_info['blocks_in_data'] = 1
        return self.slice_data_storage, data_info


class data_preprocessor_raw(data_preprocessor):
    def read_data(self):
        self.data_dir = pjoin(self.data_path,self.usrname, self.block_num)
        self.EEG_data_package = load_neuracle_data(self.data_dir)
        self.event = self.EEG_data_package['events']
        self.evnet_size = self.event.shape[0]
        self.data = self.EEG_data_package['data']
        self.sample_rate = self.EEG_data_package['srate']
        self.epoch_length = int(self.epoch_duration*self.sample_rate)
        self.visual_delay = int(0.14*self.sample_rate)
        if self.time_domain_filter == True:
            self.initial_filters()
    
    def slice_data(self,):
        if self.data_type == 'offline':
            for event_iter in range(self.evnet_size):
                event_type = self.event[event_iter][1]
                event_time_stamp = int(self.event[event_iter][0])
                epoch_cut = self.data[:,event_time_stamp+self.visual_delay:event_time_stamp+self.visual_delay+self.epoch_length+self.extended_bounds]
                epoch_cut = epoch_cut-np.mean(epoch_cut,axis=-1,keepdims=True)
                if self.time_domain_filter == True:
                    epoch_cut = self.filter_object.filter_apply(epoch_cut)
                current_event_cache = self.slice_data_storage.setdefault(event_type, list())
                current_event_cache.append(epoch_cut)
            self.event_series = self.slice_data_storage.keys()
            print('Note: The raw continuous data has been sliced by its event triggers')   
            print('Note: There are {} unique triggers in raw data'.format(len(self.event_series)))
            data_check_cache = []
            for event_iter in self.event_series:
                data_check_cache.append(len(self.slice_data_storage[event_iter]))
            data_check_cache = np.array(data_check_cache)
            unique_lens = np.unique(data_check_cache)
            if unique_lens.shape[0] == 1:
                print('Note: Data check passed, all triggers have same epochs, which is {}'.format(unique_lens[0]))
            else:
                warnings.warn('Imbalanced epochs check! The training later may go wrong, carefully check your data!')
        elif self.data_type == 'simu_online':  
            for event_iter in range(self.evnet_size):
                event_type = self.event[event_iter][1]
                event_time_stamp = int(self.event[event_iter][0])
                epoch_cut = self.data[:,event_time_stamp+self.visual_delay:event_time_stamp+self.visual_delay+self.epoch_length]
                epoch_cut = epoch_cut-np.mean(epoch_cut,axis=-1,keepdims=True)
                if self.time_domain_filter == True:
                    epoch_cut = self.filter_object.filter_apply(epoch_cut)
                self.slice_data_storage['data'].append(epoch_cut)
                self.slice_data_storage['label'].append(event_type)
    
    
# %%
def data_loader(usrname='mengqiangfan', data_path='./ThesisData/',block_num='block4'):
    data_loader = data_preprocessor(usrname, data_path, block_num)
    data_loader.read_config_file()
    data_loader.read_data()
    data_loader.slice_data()
# %%
class result_analyser():
    """ 
    A helper class for analyzing the classification results.
    """
    def __init__(self,labels,CV_loops,epoch_num,data_type='offline', *args,**kwargs): 
        """ Initial a result_analyser object

        Args:
            labels (array like vector): Triggers, which should be recorded from EEG amplifier. The triggers is a arrary which associating with how you perform your experiment.
            CV_loops (int): Numbers of CV (Cross validation) loops
            epoch_num (int): Numbers of epochs contained in single trial
        """
        self.labels = list(labels)
        self.epoch_num = epoch_num
        self.data_type = data_type
        
        if "OVERLAP" in kwargs:
            self.overlap_buffer = np.zeros((len(self.labels),kwargs["OVERLAP"]))
            self.overlap_size = kwargs["OVERLAP"]
        else:
            self.overlap_buffer = np.zeros((len(self.labels),1))
            self.overlap_size = 1
        
        if 'LDA' not in kwargs.keys():
            self.LDA_status = None
        elif kwargs['LDA'] == 'train':
                self.LDA_status = 'train'
                self.LDA_model_builder = LDA_trainer()
        elif kwargs['LDA'] == 'test':
                self.LDA_status = 'test'
                self.LDA_model = kwargs['LDA_model']
                self.verified_result_matrix = np.ones((self.epoch_num-self.overlap_size+1)) 
        
        if data_type == 'offline':
            self.trial_counter = 0
            self.CV_loops = CV_loops
            self.acc_storage = np.zeros(self.CV_loops)
            self.result_matrix = np.zeros((CV_loops,len(self.labels),self.epoch_num-self.overlap_size+1))
        elif data_type == 'simu_online':
            self.acc_storage = 0
            self.result_matrix = np.zeros((self.epoch_num-self.overlap_size+1))   
        else:
            raise Exception('Unkown value, should be \'offline\' or \'simu_online\'.')
    
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
        if self.data_type == 'offline':        
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
                
                if "LDA_model_builder" in dir(self) and self.LDA_status == 'train':
                    self.LDA_model_builder.train_data_collector(summed_coef_vector,label)    
                if self.trial_counter % len(self.labels) == 0 and epoch_iter == self.epoch_num-1:
                    self.acc_storage[CV_iter] = np.mean(self.result_matrix[CV_iter,:,:])
                    print('ACC of the {} cross validation loop is: {}'.format(CV_iter,self.acc_storage[CV_iter]))            
        elif self.data_type == 'simu_online':
            self.overlop_trials(coef_vector,epoch_iter)
            if epoch_iter >= self.overlap_size-1:
                summed_coef_vector = np.sum(self.overlap_buffer,axis=1)
                _result = self.labels[np.argmax(summed_coef_vector)]
                if _result == trial_iter:
                    self.result_matrix[epoch_iter-self.overlap_size+1] = 1
                
                if self.LDA_status == 'test':
                    pred_label = self.LDA_model.test_model(summed_coef_vector)
                    if pred_label == 0:
                        self.verified_result_matrix[epoch_iter-self.overlap_size+1] = -1

                if epoch_iter == self.epoch_num-1:
                    if self.LDA_status == None:
                        print('Note: LDA was not applied to verify the result, this is the REAL accuracy')
                        self.acc_storage = np.mean(self.result_matrix)
                        self.ACC_calculate()
                    elif self.LDA_status == 'test':
                        self.confusing_matrix_calculate()
        
    def ACC_calculate(self):
        """
        Callable function to calculate the accuracy among all cross validation loops.
        """
        self.overall_ACC = np.mean(self.acc_storage)
        print('Overall ACC of current data is {}'.format(self.overall_ACC))
    
    def confusing_matrix_calculate(self):
        negative_samples_index = np.where(self.verified_result_matrix == 0)
        positive_samples_index = np.where(self.verified_result_matrix == 1)
        negative_samples = self.result_matrix[negative_samples_index]
        positive_samples = self.result_matrix[positive_samples_index]
        true_positive = np.sum(positive_samples)/positive_samples.shape[0]
        false_positive = (positive_samples.shape[0]-np.sum(positive_samples))/positive_samples.shape[0]
        false_negative = (negative_samples.shape[0]-np.sum(negative_samples))/negative_samples.shape[0]
        true_negative = np.sum(negative_samples)/negative_samples.shape[0]
        print('NOTE: LDA was applied to verify the result.')
        print('NOTE: This is confusing matrix:')
        print('TP:{}\tTN:{}'.format(true_positive, true_negative))
        print('FP:{}\tFN:{}'.format(false_positive, false_negative))
        print('Precision:{}\tRecall:{}'.format(true_positive/(true_positive+false_negative),true_positive/(true_positive+false_negative))) 