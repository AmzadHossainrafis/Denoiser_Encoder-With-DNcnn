   
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os 



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