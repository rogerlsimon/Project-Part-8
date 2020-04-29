# Importing packages
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
import json
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pickle 

name_of_mp4_video = 'samplevid_3' # exclude mp4 extension
frame_rate = 24
num_members = 17

# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models/model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = np.concatenate((stackX, yhat), axis=1)
	# flatten predictions to [rows, members x probabilities]
	#stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX


# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat



# load dataset
dataset_X = loadtxt('outfile_X_'+str(name_of_mp4_video)+'.txt', delimiter=" ")
dataset_y = loadtxt('outfile_y_'+str(name_of_mp4_video)+'.txt', delimiter=" ")

# load all sub-models
members = load_all_models(num_members)
print('Loaded %d models' % len(members))

# load meta-learner model
pkl_filename = 'models/metamodel_'+str(num_members)+'.pkl'
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

# Predicting the Test set results
y_pred = stacked_prediction(members, model, dataset_X) # Probability of emotion shown 
accuracy = accuracy_score(dataset_y, y_pred)

#Plotting test results
seconds = np.linspace(0, len(dataset_X)/frame_rate, len(dataset_X))
fig = plt.figure()
plt.plot(seconds, y_pred)
plt.xlabel('Time [s]')
plt.ylabel('Detection [0 - No, 1 - Yes]')
plt.title('Neural network output for detecting confusion in \n' + str(name_of_mp4_video) +' video using Meta-learner')
fig.savefig('Meta-learner prediction for ' +str(name_of_mp4_video)+ ' video.tif')

# Making the Confusion Matrix (This validates the model)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dataset_y, y_pred)


confusion_detection = np.stack((seconds, y_pred), axis=-1)
json_data = {
    'confusion': confusion_detection.tolist(),
   
 }
with open('timeLabel_'+str(name_of_mp4_video)+'.json', 'w') as fp:
    json.dump(json_data, fp)
