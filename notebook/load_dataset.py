import numpy as np
import matplotlib.pyplot as plt
import random
import os

import mne
from mne.io import read_raw_ctf
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def load_data(path):
	data_path=path
	data=mne.io.read_raw_ctf(data_path , preload=True)
	sampling_rate = data.info['sfreq']
	return data , sampling_rate

def get_data_info():
	######## HARD CODED!!! ########
	file_path = os.path.join(os.getcwd(), '..', 'data/anonepi_02.ds')
	data, sampling_rate=load_data(file_path)
	#data_preprocessed = preprocessing(data)
	return data

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

def preprocessing(raw, b_drop_channels):
	picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, exclude='bads')
	raw.filter(3, 70, picks=picks,l_trans_bandwidth='auto', filter_length='auto', phase='zero', fir_design='firwin')
	raw.notch_filter(60, picks=picks, fir_design='firwin')
	# channels = ['BG1-2511', 'BG2-2511', 'BG3-2511', 'BP1-2511', 'BP2-2511', 'BP3-2511', 'BQ1-2511', 'BQ2-2511', 'BQ3-2511', 'BR1-2511', 'BR2-2511', 'BR3-2511', 'G12-2511', 'G13-2511', 'G23-2511', 'P12-2511', 'P13-2511', 'Q12-2511', 'Q13-2511', 'Q21-2511', 'Q23-2511', 'R12-2511', 'R13-2511', 'R23-2511', 'SCLK01-177', 'G11-2511', 'G22-2511', 'P11-2511', 'P22-2511', 'Q11-2511', 'Q22-2511', 'R11-2511', 'R22-2511', 'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'EEG020', 'EEG021', 'EKG']
	if b_drop_channels:
		channels = ['BG1-2511', 'BG2-2511', 'BG3-2511', 'BP1-2511', 'BP2-2511', 'BP3-2511', 'BQ1-2511', 'BQ2-2511', 'BQ3-2511', 'BR1-2511', 'BR2-2511', 'BR3-2511', 'G12-2511', 'G13-2511', 'G23-2511', 'P12-2511', 'P13-2511', 'Q12-2511', 'Q13-2511', 'Q21-2511', 'Q23-2511', 'R12-2511', 'R13-2511', 'R23-2511', 'SCLK01-177', 'G11-2511', 'G22-2511', 'P11-2511', 'P22-2511', 'Q11-2511', 'Q22-2511', 'R11-2511', 'R22-2511', 'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'EEG020', 'EEG021', 'EKG']
		raw = raw.drop_channels(channels)
	return raw

def addchannel(raw):
	info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
	stim_data = np.zeros((1, len(raw.times)))
	stim_raw = mne.io.RawArray(stim_data, info)
	raw.add_channels([stim_raw], force_update_info=True)
	return raw

def load_data_events(ds_name, window_size, b_drop_channels):
	file_path = os.path.join(os.getcwd(), '..', 'data/'+ds_name)
	data, sampling_rate=load_data(file_path)
	data_preprocessed = preprocessing(data, b_drop_channels)

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

def load_all_data_from_list(ds_list, window_size, b_drop_channels):
	epochs_array = []
	labels_array = []
	
	for ds_file in ds_list:
		print('Start loading data...')
		print('From dataset: ' + ds_file)
		data, data_sti = load_data_events(ds_file, window_size, b_drop_channels)
		epochs, labels = compute_epochs_events(data_sti, data, ds_file.split('.')[0], window_size)
		epochs_data = epochs.get_data()
		epochs_array += epochs_data.tolist()
		labels_array += labels.tolist()
		
	return epochs_array, labels_array

def load_events(window_size, b_drop_channels):
	ds_list = ['anonepi_02.ds', 'anonepi_03.ds', 'anonepi_04.ds', 'anonepi_05.ds', 'anonepi_06.ds', 'anonepi_08.ds', 'anonepi_09.ds']
	epochs_array, labels_array = load_all_data_from_list(ds_list, window_size, b_drop_channels)
	return epochs_array, labels_array

def load_dataset(window_size, test_size, random_state, b_drop_channels):
	epochs_array, labels_array = load_events(window_size, b_drop_channels)
	train_x_orig, test_x_orig, train_y_orig, test_y_orig = train_test_split(epochs_array, labels_array, test_size=test_size, random_state=random_state)
	return train_x_orig, test_x_orig, train_y_orig, test_y_orig

def norm_and_reshape(train_x_array, test_x_array):
    train_x_norm = train_x_array.reshape(train_x_array.shape[0], train_x_array.shape[1] * train_x_array.shape[2])
    test_x_norm = test_x_array.reshape(test_x_array.shape[0], test_x_array.shape[1] * test_x_array.shape[2])
    train_x = normalize(train_x_norm, norm='l2')
    test_x = normalize(test_x_norm, norm='l2')
    return train_x, test_x

def norm_and_reshape_train(train_x_array):
    train_x_norm = train_x_array.reshape(train_x_array.shape[0], train_x_array.shape[1] * train_x_array.shape[2])
    train_x = normalize(train_x_norm, norm='l2')
    return train_x

def get_train_test_data(window_size, test_size, random_state):
	train_x_orig, test_x_orig, train_y_orig, test_y_orig = load_dataset(window_size=window_size, test_size=test_size, random_state=random_state, b_drop_channels=True)
	train_x_array = np.asarray(train_x_orig)
	test_x_array = np.asarray(test_x_orig)
	train_y_array = np.asarray(train_y_orig)
	test_y_array = np.asarray(test_y_orig)
	m_train = train_x_array.shape[0]
	num_px = train_x_array.shape[1]
	m_test = test_x_array.shape[0]

	train_x, test_x = norm_and_reshape(train_x_array, test_x_array)
	return train_x, train_y_array, test_x, test_y_array

def get_trained_classifier(classfier_name, window_size, test_size, random_state):
	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
		 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
		 "Naive Bayes", "QDA"]
	classifiers = [
		KNeighborsClassifier(3),
		SVC(kernel="linear", C=0.025),
		SVC(gamma=2, C=1),
		GaussianProcessClassifier(1.0 * RBF(1.0)),
		DecisionTreeClassifier(max_depth=5),
		RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
		MLPClassifier(alpha=1),
		AdaBoostClassifier(),
		GaussianNB(),
		QuadraticDiscriminantAnalysis()]
	classifier = classifiers[names.index(classfier_name)]

	train_x, train_y_array, test_x, test_y_array = get_train_test_data(window_size, test_size, random_state)
	classifier.fit(train_x, train_y_array)
	return classifier

def get_evoked(classfier_name, window_size, test_size, random_state):
	classifier = get_trained_classifier(classfier_name, window_size, test_size, random_state)
	epochs_array, labels_array = load_events(window_size, b_drop_channels=False)
	epochs_to_array = np.asarray(epochs_array)
	norm_epochs = norm_and_reshape_train(np.asarray(epochs_to_array))
	prediction = classifier.predict(norm_epochs)
	epochs_positive = epochs_to_array[np.where(prediction==1)]
	evoked = []
	for epoch_positive in epochs_positive:
		########### HARD CODED!!!! ###########
		evoked.append(mne.EvokedArray(epoch_positive, mne.create_info(151, 600, ch_types='mag')))
	return evoked
