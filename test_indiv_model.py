# Importing packages
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from keras.models import load_model
from sklearn.metrics import accuracy_score



name_of_mp4_video = 'samplevid_5' # exclude mp4 extension
frame_rate = 24
model_number = 1

filename = 'models/model_' + str(model_number) + '.h5'
# load model from file
model = load_model(filename)

# load dataset
dataset_X = loadtxt('outfile_X_'+str(name_of_mp4_video)+'.txt', delimiter=" ")
dataset_y = loadtxt('outfile_y_'+str(name_of_mp4_video)+'.txt', delimiter=" ")

# evaluates loaded model 
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Predicting the Test set results
y_pred = model.predict(dataset_X) # Probability of emotion shown 
y_pred = (y_pred > 0.50) # True/False 
accuracy = accuracy_score(dataset_y, y_pred)
y_pred = np.reshape(y_pred, (len(y_pred),))
seconds = np.linspace(0, len(dataset_X)/frame_rate, len(dataset_X))
fig = plt.figure()
plt.plot(seconds, y_pred)
plt.xlabel('Time [s]')
plt.ylabel('Detection [0 - No, 1 - Yes]')
plt.title('Neural network output for detecting confusion in \n' + str(name_of_mp4_video)+' video using model ' +str(model_number))
fig.savefig('Model ' +str(model_number)+ ' prediction for ' +str(name_of_mp4_video)+ ' video.tif')

# Making the Confusion Matrix (This validates the model)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dataset_y, y_pred)


#confusion_detection = np.stack((seconds, y_pred), axis=-1)
#json_data = {
#    'confusion': confusion_detection.tolist(),
#   
# }
#with open('timeLabel_'+str(name_of_mp4_video)+'.json', 'w') as fp:
#    json.dump(json_data, fp)