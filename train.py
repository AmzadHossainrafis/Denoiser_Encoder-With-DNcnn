import numpy as np
from model import DnCNN
import tensorflow as tf

from dataloader import DataLoader
from utils import download_data, plot_loss, SelectCallbacks,read_config


x_train, x_test = download_data()                           # download data


config=read_config()                                        #import config 
data_loader = DataLoader(x_train=x_train,batch_size=256)    #import data loader
val_loader = DataLoader(x_train=x_test,batch_size=256)      #import validation data loader

call_backs = SelectCallbacks()                              #import callbacks
callbacks=call_backs.get_callbacks()

model=DnCNN()                                              #import model
model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=['mse'])
history1=model.fit(data_loader, epochs=config['epochs'], validation_data=val_loader,callbacks=callbacks,shuffle=config['shuffle'])

plot_loss(history1)
