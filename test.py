import tensorflow as tf
from tensorflow.keras.models import load_model


from tensorflow.keras.models import load_model
from utils import read_config,download_data

from dataloader import DataLoader


x_train, x_test = download_data()

config=read_config()

val_dataloder=DataLoader(x_train=x_test,batch_size=256)
model=load_model(config['model_weights'])
model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=['mse'])

model.evaluate(val_dataloder)