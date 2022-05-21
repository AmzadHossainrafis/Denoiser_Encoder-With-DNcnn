import numpy as np
from model import DnCNN
import tensorflow as tf
from gc import callbacks
from dataloader import DataLoader
from utils import plot_img, download_data, plot_loss, SelectCallbacks
from tensorflow.keras.callbacks import  ModelCheckpoint,CSVLogger

x_train, x_test = download_data()

data_loader = DataLoader(x_train=x_train,batch_size=256)
val_loader = DataLoader(x_train=x_test,batch_size=256)

call_backs = SelectCallbacks()
callbacks=call_backs.get_callbacks()

model=DnCNN()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
history1=model.fit(data_loader, epochs=10, validation_data=val_loader,callbacks=callbacks,shuffle=True)


