import numpy as np
from model import DnCNN
import tensorflow as tf

from dataloader import DataLoader
from utils import download_data, plot_loss, SelectCallbacks,read_config


x_train, x_test = download_data()


config=read_config()
data_loader = DataLoader(x_train=x_train,batch_size=256)
val_loader = DataLoader(x_train=x_test,batch_size=256)

call_backs = SelectCallbacks()
callbacks=call_backs.get_callbacks()

model=DnCNN()
model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=['mse'])
history1=model.fit(data_loader, epochs=config['epochs'], validation_data=val_loader,callbacks=callbacks,shuffle=config['shuffle'])

plot_loss(history1)
