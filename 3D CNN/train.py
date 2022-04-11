import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv3D, MaxPooling3D, Flatten
from keras.utils import np_utils
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

path_data = "3D CNN\\Data"
path_model = "3D CNN\\Checkpoint" 
classes = ["Pick","Stitch","Person"]
positions = ["Left"]

def load_data() : 
    videos = []
    labels = []
    for cls in classes : 
        for pos in positions : 
            for file in os.listdir(f"{path_data}\\{cls}\\{pos}") :
                vid = np.load(os.path.join(f"{path_data}\\{cls}\\{pos}",file))
                videos.append(vid)
                labels.append(classes.index(cls))
    train_videos,test_videos,train_labels,test_labels = train_test_split(videos,labels,test_size=0.2,random_state=42)
    train_videos = np.array(train_videos,dtype=np.float32)
    test_videos = np.array(test_videos,dtype=np.float32)
    train_videos = train_videos.reshape(train_videos.shape[0], 8, 64, 64, 1)
    test_videos = test_videos.reshape(test_videos.shape[0], 8, 64, 64, 1) 
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return (train_videos, train_labels),(test_videos, test_labels)

(train_videos, train_labels),(test_videos, test_labels) = load_data()

n_classes = 3
train_labels = np_utils.to_categorical(train_labels, n_classes)
test_labels = np_utils.to_categorical(test_labels, n_classes)

def make_model() :

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv3D(8, kernel_size=(3,3,3), activation='relu', input_shape=(8,64,64,1)))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(1,1,1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model

model = make_model()

print(model.summary())

model.fit(train_videos, train_labels, batch_size=32, epochs=50, validation_data=(test_videos, test_labels))

model.save(f'{path_model}\\model')

