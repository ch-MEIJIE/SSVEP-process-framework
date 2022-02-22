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

class LDA_trainer():
    def __init__(self):
        self.train_data = list()
        self.labels = list()
    
    def train_data_collector(self, coef_vector, label):
        max_data = np.max(coef_vector)
        label_predict = np.argmax(coef_vector)
        coef_vector = np.delete(coef_vector,label_predict)
        submax_data = np.max(coef_vector)
        self.train_data.append([max_data,submax_data])
        self.labels.append(label)
    
    def train_model(self):
        self.linear_model = LinearDiscriminantAnalysis()
        self.linear_model.fit(np.array(self.train_data),np.array(self.labels))

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
        self._filter_design()
    
    def _filter_design(self):
        """
        Private method of this class, it would be automatic called by init function when an new filter is created.
        Also, you can manually Design the filter here. Current version support both FIR and IIR filters design.
        """
        # Nyquist rate of signal
        nyq_rate = self.sample_rate/2.0
        # 1-D array cut of
        freq_pass = [self.low_cut_frequency/nyq_rate, self.high_cut_frequency/nyq_rate]
        freq_stop = [freq_pass[0]-2/nyq_rate,freq_pass[1]+6/nyq_rate]
        if self.filter_type == 'FIR':
            if self.filter_order%2 == 0:
                self.filter_order+=1
            # Get the fir filter coef
            taps = scipy.signal.firwin(self.filter_order, freq_pass, window='hamming', pass_zero='bandpass')
            self.filter_b = taps
            self.filter_a = 1
        elif self.filter_type == 'IIR':
            ord, wn = scipy.signal.cheb1ord(freq_pass,freq_stop,4,30)
            self.filter_b,self.filter_a = scipy.signal.cheby1(ord,0.5,wn,btype='bandpass')
            
    def filter_apply(self,data):
        """Callable function for filtering data.
        Use this function when you need to filter data after the class object is created.

        Args:
            data (numpy array): The data to be filtered. Shape as n_chan

        Returns:
            numpy array: filterd data, same shape as input data.
        """
        filter_data = scipy.signal.filtfilt(self.filter_b,self.filter_a,data,)
        return filter_data

# %%
# Test filter
filter = filter_applyer(250,6,80,48,'FIR')

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
    
    def __init__(self, usrname, data_path, block_num, file_name='EEG.mat', time_domain_filter = True, extended_bounds = 0):
        self.usrname = usrname
        self.data_path = data_path
        self.block_num = block_num
        self.file_name = file_name
        self.slice_data_storage = dict()
        self.time_domain_filter = time_domain_filter
        self.extended_bounds = extended_bounds
    
    def read_config_file(self, file_name = 'config.json'):
        with open(file_name) as file:
            self.data_config = json.load(file)
        self.blocks_in_data = self.data_config['blocks_in_data']
        self.epochs_in_data = self.data_config['epochs_in_trials']
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
    
    def initial_filters(self):
        self.filter_object = filter_applyer(self.sample_rate, low_cut_frequency=7.0, high_cut_frequency=80.0, filter_order=64, filter_type='IIR')
    
    def slice_data(self,):
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
    
    def send_data(self):
        return self.slice_data_storage

# %%
def data_loader(usrname='mengqiangfan', data_path='./ThesisData/',block_num='block4'):
    data_loader = data_preprocessor(usrname, data_path, block_num)
    data_loader.read_config_file()
    data_loader.read_data()
    data_loader.initial_filters()
    data_loader.slice_data()

# %%
if __name__ == '__main__':
    data_loader()
# %%
