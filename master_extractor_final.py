# Importing packages
from mlxtend.image import extract_face_landmarks
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model

len_of_feature_vec = 4096 

# load model without classifier layers
model = VGG16()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

newsize = (224, 224) 

num_of_cols = len_of_feature_vec + 68*2

# LANDMARK GENERATION 

name_of_mp4_video = 'samplevid_9' # exclude mp4 extension
frame_rate = 24 
start_of_emotion = 18 # This value is provided in seconds
end_of_emotion =  25 # This value is provided in seconds 



def extractFrames(pathIn, model):
    cap = cv2.VideoCapture(pathIn)
    X = np.zeros((1,num_of_cols)) 
    i = 1
 
    while (cap.isOpened()):
 
        # Capture frame-by-frame 
        ret, frame = cap.read()

 
        if ret == True:
            
            # Extracting and saving facial landmarks from each frame
            frame = np.swapaxes(frame, 0, 1) # This rotates the image back to appropriate view             
    
        
# Facial landmark extractor
            try:
                landmarks = extract_face_landmarks(frame)
            except:
                print('No face detected!')
                landmarks = -1*np.ones((68,2))
            else:
                landmarks = landmarks.flatten()[np.newaxis]
            finally:
                print('Facial detection algorithm is done!')
                
# VGG16 feature extractor 
            # convert the image pixels to a numpy array
            frame = cv2.resize(frame, newsize) # size image down by (224, 224) for VGG16
            image = img_to_array(frame)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            # get extracted features
            feature_vector = model.predict(image)
            print("VGG16's feature vector extracted!")
            
            X_new = np.append(landmarks, feature_vector, axis=1)
            print('Frame '+str(i)+' : Concatenation of row vectors is complete!')
            X = np.concatenate((X, X_new))
            i += 1
        else:
            X = np.delete(X, (0), axis=0) # Deletes entire first row because it was used for initialization
            np.savetxt('outfile_landmarks_'+str(name_of_mp4_video)+'.txt', X, fmt= '%.7f', delimiter= ' ') # Saves non-scaled facial landmarks 
                                                                            # for each frame to text file in case it might be needed later 
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
            break
    return X

# Import videos and read frame by frame to extract facial landmarks but don't save images to a folder (saves space)
print('Extracting facial landmarks and feature vectors frame-by-frame...')
X = extractFrames(str(name_of_mp4_video)+'.mp4', model)
print('Extraction complete.')
y = np.zeros((len(X), 1))

# Feature Scaling (required to train a DNN)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_final = sc.fit_transform(X)
np.savetxt('outfile_X_'+str(name_of_mp4_video)+'.txt', X_final, fmt= '%.7f', delimiter= ' ') # Saves scaled facial landmarks and feature vectors    
                                                                        # for each frame to text file in case it might be needed later 

# Replace 0's with 1's where the emotion occurs
frame_location1 = frame_rate*start_of_emotion
frame_location2 = frame_rate*end_of_emotion
y[frame_location1-1:frame_location2,] = 1
np.savetxt('outfile_y_'+str(name_of_mp4_video)+'.txt', y, fmt= '%.0f', delimiter= ' ') # Saves facial landmarks for each frame 
                                                                                    # to text file in case it might be needed later 
