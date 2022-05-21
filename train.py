from gc import callbacks
from tabnanny import check
from dataloader import DataLoader
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model import DnCNN
from utils import plot_img, download_data, plot_loss
#from sklearn.preprocessing import train_test_split
from tensorflow.keras.callbacks import  ModelCheckpoint,CSVLogger

x_train, x_test = download_data()

data_loader = DataLoader(x_train=x_train,batch_size=256)
val_loader = DataLoader(x_train=x_test,batch_size=256)
check_point=ModelCheckpoint(filepath="logs/model_weights/weights_dncnn.hdf5",save_best_only=True,monitor='val_loss')
CSV=CSVLogger(filename="logs/csv_logs/log.csv")

callbacks=[check_point,CSV]

model=DnCNN()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history1=model.fit(data_loader, epochs=10, validation_data=val_loader,callbacks=callbacks,shuffle=True)


