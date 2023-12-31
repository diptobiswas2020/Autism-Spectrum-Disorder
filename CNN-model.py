import os
print(os.listdir('/kaggle/input/asd-cnn-data/data'))

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image

X=[]
Z=[]
IMG_SIZE=150
CONSOLIDATED_DIR_AUTISTIC='/kaggle/input/asd-cnn-data/data/consolidated/Autistic'
CONSOLIDATED_DIR_NONAUTISTIC='/kaggle/input/asd-cnn-data/data/consolidated/Non_Autistic'
TEST_DIR_AUTISTIC='/kaggle/input/asd-cnn-data/data/test/Autistic'
TEST_DIR_NONAUTISTIC='/kaggle/input/asd-cnn-data/data/test/Non_Autistic'
#TRAIN_DIR='/kaggle/input/asd-cnn-data/data/train'
TRAIN_DIR_AUTISTIC='/kaggle/input/asd-cnn-data/data/train/Autistic'
TRAIN_DIR_NONAUTISTIC='/kaggle/input/asd-cnn-data/data/train/Non_Autistic'
VALID_DIR_AUISTIC='/kaggle/input/asd-cnn-data/data/valid/Autistic'
VALID_DIR_NONAUISTIC='/kaggle/input/asd-cnn-data/data/valid/Non_Autistic'

def assign_label(img,asd_type):
    return asd_type
	
def make_train_data(asd_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,asd_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        #img = cv2.resize(img, dsize=(IMG_SIZE,IMG_SIZE),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        
        X.append(np.array(img))
        Z.append(str(label))
		
make_train_data('TRAIN_DIR_AUTISTIC',TRAIN_DIR_AUTISTIC)
print(len(X))

make_train_data('TRAIN_DIR_NONAUTISTIC',TRAIN_DIR_NONAUTISTIC)
print(len(X))

make_train_data('TEST_DIR_NONAUTISTIC',TEST_DIR_NONAUTISTIC)
print(len(X)

make_train_data('TEST_DIR_AUTISTIC',TEST_DIR_AUTISTIC)
print(len(X))

fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('ASD: '+Z[l])
        
plt.tight_layout()

le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

np.random.seed(42)
rn.seed(42)
#tf.set_random_seed(42)
tf.random.set_seed(42)

# # modelling starts using a CNN.
#Building the ConvNet Model

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))


#Using a LR Annealer

batch_size=128
epochs=50
#epochs = 2

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

# data augmentation

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

History = model.fit(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
							  
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

