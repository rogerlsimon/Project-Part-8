# Importing packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
from os import makedirs
from numpy import loadtxt
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import load_model
from sklearn.linear_model import LogisticRegression
import pickle
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD


# fit model on dataset
def fit_model(trainX, trainy, testX, testy):
    classifier = Sequential() # Name of ANN
    classifier.add(Dense(units = 140, kernel_initializer = 'uniform', activation = 'relu', input_dim = trainX.shape[1])) 
    classifier.add(Dropout(0.2)) 
    classifier.add(Dense(units = 140, kernel_initializer = 'uniform', activation = 'relu')) 
    classifier.add(Dropout(0.2))          
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 
    opt = SGD(lr = 0.0001)
    classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy']) 
    history = classifier.fit(trainX, trainy, batch_size = 80, epochs = 500, validation_data = (testX, testy), shuffle=True)
    return classifier, history 


 
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

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy, num_members):
	# create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
	# fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    pkl_filename = 'models/metamodel_'+str(num_members)+'.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    return model
 
# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat
 

X_train = loadtxt('outfile_X_confusion_dataset.txt', delimiter=" ")
X_train = X_train
X_new = loadtxt('outfile_X_samplevid_3.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))
X_new = loadtxt('outfile_X_samplevid_4.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))
X_new = loadtxt('outfile_X_samplevid_5.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))
X_new = loadtxt('outfile_X_samplevid_6.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))
X_new = loadtxt('outfile_X_samplevid_9.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))
X_new = loadtxt('outfile_X_samplevid_10.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))
X_new = loadtxt('outfile_X_samplevid_12.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))
X_new = loadtxt('outfile_X_samplevid_15.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))
X_new = loadtxt('outfile_X_samplevid_16.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))
X_new = loadtxt('outfile_X_samplevid_17.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))
X_new = loadtxt('outfile_X_samplevid_18.txt', delimiter=" ")
X_new = X_new
X_train = np.concatenate((X_train, X_new))

X_test = loadtxt('outfile_X_samplevid_2.txt', delimiter=" ")
X_test = X_test
X_new = loadtxt('outfile_X_samplevid_7.txt', delimiter=" ")
X_new = X_new
X_test = np.concatenate((X_test, X_new))
X_new = loadtxt('outfile_X_samplevid_8.txt', delimiter=" ")
X_new = X_new
X_test = np.concatenate((X_test, X_new))
X_new = loadtxt('outfile_X_samplevid_11.txt', delimiter=" ")
X_new = X_new
X_test = np.concatenate((X_test, X_new))
X_new = loadtxt('outfile_X_samplevid_13.txt', delimiter=" ")
X_new = X_new
X_test = np.concatenate((X_test, X_new))
X_new = loadtxt('outfile_X_samplevid_14.txt', delimiter=" ")
X_new = X_new
X_test = np.concatenate((X_test, X_new))

y_train = loadtxt('outfile_y_confusion_dataset.txt', delimiter=" ")
y_new = loadtxt('outfile_y_samplevid_3.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))
y_new = loadtxt('outfile_y_samplevid_4.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))
y_new = loadtxt('outfile_y_samplevid_5.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))
y_new = loadtxt('outfile_y_samplevid_6.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))
y_new = loadtxt('outfile_y_samplevid_9.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))
y_new = loadtxt('outfile_y_samplevid_10.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))
y_new = loadtxt('outfile_y_samplevid_12.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))
y_new = loadtxt('outfile_y_samplevid_15.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))
y_new = loadtxt('outfile_y_samplevid_16.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))
y_new = loadtxt('outfile_y_samplevid_17.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))
y_new = loadtxt('outfile_y_samplevid_18.txt', delimiter=" ")
y_train = np.concatenate((y_train, y_new))

y_test = loadtxt('outfile_y_samplevid_2.txt', delimiter=" ")
y_new = loadtxt('outfile_y_samplevid_7.txt', delimiter=" ")
y_test = np.concatenate((y_test, y_new))
y_new = loadtxt('outfile_y_samplevid_8.txt', delimiter=" ")
y_test = np.concatenate((y_test, y_new))
y_new = loadtxt('outfile_y_samplevid_11.txt', delimiter=" ")
y_test = np.concatenate((y_test, y_new))
y_new = loadtxt('outfile_y_samplevid_13.txt', delimiter=" ")
y_test = np.concatenate((y_test, y_new))
y_new = loadtxt('outfile_y_samplevid_14.txt', delimiter=" ")
y_test = np.concatenate((y_test, y_new))



trainX = X_train
testX = X_test
trainy = y_train
testy = y_test

# create directory for models
makedirs('models')
# fit and save models
n_members = 10
for i in range(n_members):
	# fit model
    model, history = fit_model(trainX, trainy, testX, testy)
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model '+str(i+1) + ' accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig('model_'+str(i+1) + '_accuracy.tif')
    # summarize history for loss
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model '+str(i+1) + ' loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig('model_'+str(i+1) + '_loss.tif')
	# save model
    filename = 'models/model_' + str(i + 1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)

classifier = model
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
for model in members:
	_, acc = model.evaluate(testX, testy, verbose=0)
	print('Model Accuracy: %.3f' % acc)
# fit stacked model using the ensemble
i = 1

while i != n_members:
    model = fit_stacked_model(members[0:i+1], trainX, trainy, i+1)
    # evaluate model on test set
    yhat = stacked_prediction(members[0:i+1], model, testX)
    if i == 1:
        acc = accuracy_score(testy, yhat)
        print('Stacked Test Accuracy: %.3f' % acc)
        i +=1
    else: 
        acc_new = accuracy_score(testy, yhat)
        print('Stacked Test Accuracy: %.3f' % acc_new)
        acc = np.append(acc, acc_new)
        i +=1
# summarize history for accuracy
fig = plt.figure()
plt.plot(np.arange(n_members - 1) + 2, acc)
plt.title('Stacked Test Accuracy vs. Number of individual models used')
plt.ylabel('Stacked Test Accuracy')
plt.xlabel('Number of models')
plt.show()
fig.savefig('stackedmodelaccuracyfor_numofmodels.tif') 

plot_model(classifier, to_file='model_plot_for.png', show_shapes=True, show_layer_names=True)    
