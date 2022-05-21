   
from tkinter.tix import IMMEDIATE
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os 
import math
import yaml
from tensorflow import keras



# data_loader = DataLoader(batch_size=256)
# val_loader = DataLoader(x_train=x_test,batch_size=256)
# no_noise_img = model.predict(val_loader)
# x,y=data_loader[0]

def plot_img(no_noise_img):
    plt.figure(figsize=(40, 4))
    for i in range(10):
        # display original
        # display reconstructed (after noise removed) image
        ax = plt.subplot(3, 20, 40 +i+ 1)

        plt.imshow(no_noise_img[i].reshape(28, 28), cmap="binary")

    plt.show()

def download_data():
    """
    download the data and normalize it

    """
    (x_train, _), (x_test,_) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    return x_train, x_test


def plot_loss(history):
    """
    plot the loss history
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(r"C:\Users\Amzad\Desktop\keras_project\denoiser_encoder\logs\Prediction\loss.png")
    plt.show()


def make_dir(path):
    """
    make a directory
    """
    dir_list= os.listdir(path)
    if "logs" not in dir_list:
        os.mkdir(path+"logs")
        os.mkdir(path+"logs/csv_logs")
        os.mkdir(path+"logs/prediction")
        os.mkdir(path+"logs/model_weights")
    else:
        print("logs directory already exists")


def read_config(path=r"C:\Users\Amzad\Desktop\keras_project\denoiser_encoder\config.yaml"):
    """
    read the config file
    """
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config



# Callbacks and Prediction during Training
# ----------------------------------------------------------------------------------------------
class SelectCallbacks(keras.callbacks.Callback):
    def __init__(self,config= read_config()):
        """
        Summary:
            callback class for validation prediction and create the necessary callbacks objects
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model object
            config (dict): configuration dictionary
        Return:
            class object
        """
        super(keras.callbacks.Callback, self).__init__()
        self.config = config
        self.callbacks = []

    def lr_scheduler(self, epoch):
        """
        Summary:
            learning rate decrease according to the model performance
        Arguments:
            epoch (int): current epoch
        Return:
            learning rate
        """
        drop = 0.5
        epoch_drop = self.config['epochs'] / 8.
        lr = self.config['learning_rate'] * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
        return lr


    def get_callbacks(self):
        """
        Summary:
            creating callbacks based on configuration
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model class object
        Return:
            list of callbacks
        """
        if self.config['csv']:
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(self.config['csv_log_dir'], self.config['csv_log_name']), separator = ",", append = False))
        
        if self.config['checkpoint']:
            self.callbacks.append(keras.callbacks.ModelCheckpoint(filepath=self.config['checkpoint_dir']+"weights_dncnn.hdf5", save_best_only = True))
        if self.config['lr']:
            self.callbacks.append(keras.callbacks.LearningRateScheduler(schedule = self.lr_scheduler))
        
        
        
        return self.callbacks
