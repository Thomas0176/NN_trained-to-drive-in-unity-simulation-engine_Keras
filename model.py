
# coding: utf-8

# In[1]:

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import csv
import matplotlib.pyplot as plt
import math

            
example_img_filename = "data/IMG/center_2016_12_01_13_30_48_287.jpg"
train_data_folder = "steering_data/IMG/"
csv_filename = "steering_data/edit_driving_log.csv"
train_images_filenames = os.listdir(train_data_folder)

example_img = np.asarray(Image.open(example_img_filename).convert('RGB'))
print(example_img.shape)


# In[ ]:

# reading the X_images
    
# for index, filename in enumerate(train_images_filenames):
#     print(filename)
#     print("Reading in the x data...: ", index)
#     image = np.asarray(Image.open(train_data_folder+filename).convert('RGB'))
    
#     X_test_new_img = np.empty(shape=[1, example_img.shape[0], example_img.shape[1], example_img.shape[2]])
#     X_test_new_img = X_test_new_img.astype('uint8')
#     X_test_new_img[0] = image
#     X_train = np.vstack((X_train, X_test_new_img))


# In[ ]:




# In[ ]:

# reading the y labels
# columns: ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
# first row = headers

# y_train = np.zeros(len(X_train))
# y_train = y_train.astype('float64')
# loaded_y_column = 3 # column 'steering'

# with open(csv_filename, 'r') as f:
#     reader = csv.reader(f)
    
#     for index, row in enumerate(reader):
#         #skip first row of csv as it is just the header
        
#         if (index != 0):
#             #print(train_images_filenames[index-1])
#             y_train[index-1]=row[3]
#             assert("IMG/"+train_images_filenames[index-1] == row[0]), "Filename of y is different from filename in x"


# In[ ]:

# Splitting the dataset

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.33)


# In[ ]:

#assert imported data

# STOP: Do not change the tests below. Your implementation should pass these tests. 
# assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
# assert(X_train.shape[1:] == (160,320,3)), "The dimensions of the images are not 32 x 32 x 3."
# assert(X_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels."
# assert(X_val.shape[1:] == (160,320,3)), "The dimensions of the images are not 32 x 32 x 3."



# In[ ]:

# Preprocess the data

# TODO: Implement data normalization here.
# X_train = X_train.astype('float32')
# X_val = X_val.astype('float32')
# X_train = X_train / 255 - 0.5
# X_val = X_val / 255 - 0.5


# In[ ]:

# STOP: Do not change the tests below. Your implementation should pass these tests. 
# assert(math.isclose(np.min(X_train), -0.5, abs_tol=1e-5) and math.isclose(np.max(X_train), 0.5, abs_tol=1e-5)), "The range of the training data is: %.1f to %.1f" % (np.min(X_train), np.max(X_train))
# assert(math.isclose(np.min(X_val), -0.5, abs_tol=1e-5) and math.isclose(np.max(X_val), 0.5, abs_tol=1e-5)), "The range of the validation data is: %.1f to %.1f" % (np.min(X_val), np.max(X_val))


# # Building the neural network with Keras here.

# In[38]:

from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Convolution2D, Flatten, Dropout

#based on NVIDIA selfdriving car network: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

batch_size = 128
nb_epoch = 20

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_val = np_utils.to_categorical(y_val, nb_classes)

model = Sequential()

#input shape
input_shape = (160, 320, 3)
# number of convolutional filters to use
nb_filters_1 = 24
nb_filters_2 = 36
nb_filters_3 = 48
nb_filters_4 = 64
nb_filters_5 = 64
kernel_size_1 = (5, 5)
kernel_size_2 = (3, 3)
subsample_1=(2, 2)
subsample_2=(1, 1)



model.add(Convolution2D(nb_filters_1, kernel_size_1[0], kernel_size_1[1], subsample=subsample_1, border_mode='valid', input_shape=input_shape, activation='relu'))
model.add(Convolution2D(nb_filters_2, kernel_size_1[0], kernel_size_1[1], subsample=subsample_1, border_mode='valid', activation='relu'))
model.add(Convolution2D(nb_filters_3, kernel_size_1[0], kernel_size_1[1], subsample=subsample_1, border_mode='valid', activation='relu'))
model.add(Convolution2D(nb_filters_4, kernel_size_2[0], kernel_size_2[1], subsample=subsample_2, border_mode='valid', input_shape=input_shape, activation='relu'))
model.add(Convolution2D(nb_filters_5, kernel_size_2[0], kernel_size_2[1], subsample=subsample_2, border_mode='valid', input_shape=input_shape, activation='relu'))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['loss'])


# In[41]:

def generate_arrays_from_file(split_start, split_end):
    while 1: 
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
        
            #iterate row_count times 
            print ("Nr. of row: ", int(split_end*row_count))
            firstPass = True
            
            
            for index in range (int(split_start*row_count), int(split_end*row_count)):
                random_row_index = random.randrange(int(split_start*row_count)+1, int(split_end*row_count)) #don't use first row
#                 print ("Random row index: ", random_row_index)

                
                #get the random row;
                for i, row in enumerate(reader):
                    if i == random_row_index:
                        random_row = row
                        break

                #enter every batch size, incl. first pass
                if ((index % batch_size) == 0 or firstPass):
                    print("In batch")

                    #don't yield during first pass
                    if index != 0 and not firstPass:

                        print ("in yield with index ", index)
#                         STOP: Do not change the tests below. Your implementation should pass these tests. 
                        assert(X_data.shape[0] == y_data.shape[0]), "The number of images is not equal to the number of labels."
                        assert(X_data.shape[1:] == (160,320,3)), "The dimensions of the images are not 32 x 32 x 3."

#                       STOP: Do not change the tests below. Your implementation should pass these tests. 
                        assert(math.isclose(np.min(X_data), -0.5, abs_tol=1e-5) and math.isclose(np.max(X_data), 0.5, abs_tol=1e-5)), "The range of the training data is: %.1f to %.1f" % (np.min(X_data), np.max(X_data))
                        yield X_data, y_data
                    X_data = np.empty(shape=[batch_size, example_img.shape[0], example_img.shape[1], example_img.shape[2]])
#                     X_data = X_data.astype('uint8')
#                     X_data[None,:,:,:] = X_data[None,:,:,:].astype('uint8')
                    X_data[None,:,:,:] = X_data[None,:,:,:].astype('float32')

                    y_data = np.zeros(batch_size)
                    y_data = y_data.astype('float32')
                    index_xy = 0
                

                image = np.asarray(Image.open(train_data_folder+train_images_filenames[random_row_index-1]).convert('RGB'))
#                 image = image.astype('uint8')
                image = image.astype('float32')
                image = image/255-0.5                   
#                 X_data[None,:,:,:].astype('uint8')
                X_data[index_xy]= image
                X_data = X_data.astype('float32')
                y_data[index_xy]=random_row[3]
                firstPass = False

                
                
#                 print("X_data shape")
#                 print(X_data.shape)
#                 print("y_data")
#                 print(y_data.shape)
                
#                 print("This is the line.")
#                 print(random_row[3])
#                 print("This is the filename")
#                 print(str(train_images_filenames[random_row_index-1]))
#                 print("This is index_xy")
#                 print(index_xy)
#                 print("This is index_xy shap eof Xtest new img")
#                 print(X_data[index_xy].shape)
#                 print(type(X_data[index_xy]))
#                 print(type(image))




#                 plt.title(str(train_images_filenames[random_row_index-1]))
#                 plt.imshow(image)
#                 plt.show()
#                 time.sleep(1) # delays for 5 seconds

#                 plt.imshow(X_data[index_xy])
#                 plt.show()
                
                index_xy += 1


batch_size = 126 #128 #7936

with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader) 
        print ("Row count: ", row_count)

train_perc=0.66
val_perc = 1-train_perc

# history = model.fit_generator(generate_arrays_from_file(0,train_perc), samples_per_epoch=100, nb_epoch=1, verbose=1, validation_data=generate_arrays_from_file(val_perc, 1),
# nb_val_samples=row_count*val_perc)

history = model.fit_generator(generate_arrays_from_file(0,train_perc), samples_per_epoch=int(row_count*train_perc), nb_epoch=1, verbose=1, validation_data=generate_arrays_from_file(val_perc, 1),
nb_val_samples=row_count*val_perc)

# history = model.fit_generator(generate_arrays_from_file(), samples_per_epoch=1000, nb_epoch=2, verbose=1)

# history = model.fit_generator(generate_arrays_from_file(), samples_per_epoch=10000, nb_epoch=10, verbose=1)





# In[40]:

#SAVE THE MODEL
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...


# In[44]:

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)

print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



