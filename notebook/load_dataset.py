import numpy as np
import matplotlib.pyplot as plt
import random
import os

import mne
from mne.io import read_raw_ctf
from sklearn.model_selection import train_test_split

def load_data(path):
	data_path=path
	data=mne.io.read_raw_ctf(data_path , preload=True)
	sampling_rate = data.info['sfreq']
	return data , sampling_rate

def markers(path, sampling_rate):
	marker = []
	with open(path, 'r') as file: 
		a= file.readline()
		while a != "":
			 a= file.readline()
			 if any(c in u'+' for c in a):
					 q=a.split()
					 marker.append(int(float(q[1])*sampling_rate))
	return marker

def random_markers(window_size, sampling_rate, marker):
	random_marker = []
	buffer = window_size*sampling_rate
	if len(marker) <= 0:
		return random_marker
	for i in range(len(marker)):
		if i == 0 and marker[0] > buffer:
			random_marker.append(random.randint(buffer,marker[0]))
		elif marker[i-1]+buffer < marker[i]-buffer:
			random_marker.append(random.randint(marker[i-1]+buffer,marker[i]-buffer))
	random_marker.append(random.randint(marker[i]+buffer, 72000-buffer))
	return random_marker

def preprocessing(raw):
	picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, exclude='bads')
	raw.filter(3, 70, picks=picks,l_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_design='firwin')
	channels = ['BG1-2511', 'BG2-2511', 'BG3-2511', 'BP1-2511', 'BP2-2511', 'BP3-2511', 'BQ1-2511', 'BQ2-2511', 'BQ3-2511', 'BR1-2511', 'BR2-2511', 'BR3-2511', 'G12-2511', 'G13-2511', 'G23-2511', 'P12-2511', 'P13-2511', 'Q12-2511', 'Q13-2511', 'Q21-2511', 'Q23-2511', 'R12-2511', 'R13-2511', 'R23-2511', 'SCLK01-177', 'G11-2511', 'G22-2511', 'P11-2511', 'P22-2511', 'Q11-2511', 'Q22-2511', 'R11-2511', 'R22-2511', 'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'EEG020', 'EEG021', 'EKG']
	raw = raw.drop_channels(channels)
	return raw

def addchannel(raw):
	info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
	stim_data = np.zeros((1, len(raw.times)))
	stim_raw = mne.io.RawArray(stim_data, info)
	raw.add_channels([stim_raw], force_update_info=True)
	return raw

def load_data_events(ds_name, window_size):
	file_path = os.path.join(os.getcwd(), '..', 'data/'+ds_name)
	data, sampling_rate=load_data(file_path)
	data_preprocessed = preprocessing(data)

	# add stimulus channel
	# positive = 1, negative = 2
	data_sti = addchannel(data_preprocessed)
	marker_epi = markers(file_path+'/MarkerFile.mrk' , sampling_rate)
	marker_non = random_markers(window_size, sampling_rate, marker_epi)
	event_1 = marker_epi + marker_non
	event_2 = [0] * (len(marker_epi)+len(marker_non))
	event_3 = [1] * len(marker_epi) + [2] * len(marker_non)
	events = np.array([event_1, event_2, event_3]).T
	data_sti.add_events(events, stim_channel='STI')
	return data, data_sti

def compute_epochs_events(data_sti, raw, trial_name, window_size):
	events = mne.find_events(data_sti, stim_channel='STI')
	# Events:
	# 2 - non-epi window
	# 1 - epi window
	# picks = mne.pick_types(data.info, meg='grad', eeg=False, stim=False, eog=False, exclude='bads')
	picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False, exclude='bads')
	event_id = dict(positive=1, negative=2)
	tmin = -window_size/2  # start of each epoch (500ms before the trigger)
	tmax = window_size/2  # end of each epoch (500ms after the trigger)
	epochs = mne.Epochs(data_sti, events, event_id, tmin, tmax, proj=False, picks=picks, baseline=None, preload=True)
	epochs_proj = mne.Epochs(data_sti, events, None, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
	labels = epochs.events[:, -1]
	
	# output epochs/labels info
	print(epochs)
	#epochs.plot()
	#epochs.plot_psd(fmin=2., fmax=500)
	print(epochs.get_data().shape)
	# print(epochs.get_data())
	print(labels)

	# evoked without ssp
#     evoked = epochs.average()
#     print(evoked)
#     evoked.plot()

	epochs.save(trial_name+'epochs-epo.fif')
	return epochs, labels

def load_all_data_from_list(ds_list, window_size):
	epochs_array = []
	labels_array = []
	
	for ds_file in ds_list:
		print('Start loading data...')
		print('From dataset: ' + ds_file)
		data, data_sti = load_data_events(ds_file, window_size)
		epochs, labels = compute_epochs_events(data_sti, data, ds_file.split('.')[0], window_size)
		epochs_data = epochs.get_data()
		epochs_array += epochs_data.tolist()
		labels_array += labels.tolist()
		
	return epochs_array, labels_array

def load_dataset(window_size, test_size, random_state):
	ds_list = ['anonepi_02.ds', 'anonepi_03.ds', 'anonepi_04.ds', 'anonepi_05.ds', 'anonepi_06.ds', 'anonepi_08.ds', 'anonepi_09.ds']
	epochs_array, labels_array = load_all_data_from_list(ds_list, window_size)
	train_x_orig, test_x_orig, train_y_orig, test_y_orig = train_test_split(epochs_array, labels_array, test_size=test_size, random_state=random_state)
	return train_x_orig, test_x_orig, train_y_orig, test_y_orig
