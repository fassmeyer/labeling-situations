import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed 
import keras 
from keras import layers
from keras import backend as K
import numpy as np
from keras import losses
import matplotlib.pyplot as plt
from keras.utils import plot_model



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_encoder(hidden_dim, intermediate_dim, channels):
    tracking_frame = layers.Input(shape=(105, 68, channels))
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(tracking_frame) # -> (x_coordinate, y_coordinate, channels)
    x = layers.Conv2D(64, (3,3), padding="same", activation = "relu", strides=(2,2))(x) 
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    feature_vector = layers.Dense(300, activation="relu")(x)
    feature_dim = K.int_shape(feature_vector)[1]
    h = layers.Dense(units=intermediate_dim)(feature_vector)
    z_mean = layers.Dense(hidden_dim)(h)
    z_log_sigma = layers.Dense(hidden_dim)(h)
    z = Sampling()([z_mean, z_log_sigma])
    encoder = keras.Model(tracking_frame, [z_mean, z_log_sigma, z], name="encoder")
    return encoder


def get_decoder(hidden_dim, intermediate_dim, channels, with_sigmoid=False):
    decoder_input = layers.Input(shape=(hidden_dim,))
    f_vector = layers.Dense(units=300)(decoder_input)
    x = layers.Dense(units = 53*34*64)(f_vector)
    x = layers.Reshape(target_shape=(53, 34, 64))(x)
    x = layers.Conv2DTranspose(64, (3,3), padding="same", activation = "relu", strides=(2,2))(x)
    x = layers.Cropping2D(cropping=((1,0),(0,0)))(x)
    if with_sigmoid:
        t_frame = layers.Conv2D(channels, (3,3),  padding="same", activation='sigmoid')(x) 
    else:
        t_frame = layers.Conv2D(channels, (3,3),  padding="same")(x) 
    decoder = keras.Model(decoder_input, t_frame,  name="decoder")
    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder  
        
    def encode(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        return z_mean, z_log_var, z
    
    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction   
    











