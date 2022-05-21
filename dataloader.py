
import numpy as np
import tensorflow as tf


class DataLoader(tf.keras.utils.Sequence):
 

    def __init__(self, x_train, batch_size=32,noise_factor=.5, loc=0,shuffle=True):
        super().__init__()
        self.x_train = x_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loc = loc
        self.noise_factor = noise_factor


    def __len__(self):

        return int(np.ceil(len(self.x_train) / float(self.batch_size)))


    def __add_noise(self,image):
        """
        artguments: image(array)
        returns: image with noise
        
        """
        train_noisy =image + self.noise_factor * np.random.normal(loc=self.loc, scale=1.0, size=image.shape) 
        #x_test_noisy = x_test + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
        x_train_noisy = np.clip(train_noisy, 0., 1.)
        return x_train_noisy


       
    def __getitem__(self, idx):
        batch_x = self.x_train[idx * self.batch_size:(idx + 1) * self.batch_size] # get the batch


        tarin=np.zeros((self.batch_size,28,28,1),dtype="float32")
        lable=np.zeros((self.batch_size,28,28,1),dtype="float32")

        for indx, i in  enumerate(batch_x):
            x_train = self.__add_noise(batch_x[indx]) #add noise to the image
            x_train= np.reshape(x_train,(28,28,1))    #reshape the image to add channel
            tarin[indx]=x_train
            lable[indx]=np.reshape(batch_x[indx],(28,28,1)) #reshape the image to add channel

        return tarin,lable


  



 