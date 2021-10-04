import numpy as np
import keras
import tensorflow as tf
from keras import layers


class Sampling(layers.Layer):
  """Uses (pred_mean, pred_var) to sample z_t."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch_size = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_feature_extractor(image_0, image_1, channels, feature_dim):
    inp = layers.Input((image_0, image_1, channels))
    x = layers.Conv2D(64, (5,5), strides=(2,2), padding='same')(inp)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    out = layers.Dense(units = feature_dim)(x)
    frame_feature_extractor = keras.Model(inp, out, name="frame_feature_extractor")
    return frame_feature_extractor


def get_MLP_a(feature_dim, hidden_dim):
    inp = layers.Input((feature_dim,))
    a_mean = layers.Dense(hidden_dim)(inp)
    a_log_var = layers.Dense(hidden_dim)(inp)
    a = Sampling()([a_mean, a_log_var])
    model = keras.Model(inp, [a_mean, a_log_var, a])
    return model


def get_MLP_z(feature_dim, hidden_dim):
    inp = layers.Input((feature_dim,))
    a = layers.Input((hidden_dim,))
    h = layers.Concatenate()([inp, a])
    z_mean = layers.Dense(hidden_dim)(h)
    z_log_var = layers.Dense(hidden_dim)(h)
    z = Sampling()([z_mean, z_log_var])
    model = keras.Model([inp, a], [z_mean, z_log_var, z])
    return model

def get_decoder(image_0, image_1, hidden_dim, channels=9):
    a_input = keras.Input((hidden_dim,))
    z_input = keras.Input((hidden_dim,))
    inp = layers.Concatenate()([a_input, z_input])
    x = layers.Dense(units=128*int((1/4)*image_0)*(1/4)*image_1)(inp)
    x = layers.Reshape(target_shape=(int((1/4)*image_0),int((1/4)*image_1),128))(x)
    x = layers.Conv2DTranspose(128, (5,5), padding="same", strides=(1,1), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (5,5), padding="same", strides=(2,2), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    reconstruction = layers.Conv2DTranspose(channels, (5,5), strides=(2,2), padding="same", activation="tanh")(x)
    reconstruction = layers.Conv2D(channels, (5,5),  padding="same", activation='sigmoid')(reconstruction) 
    if image_0 % 2 != 0:
        reconstruction = layers.ZeroPadding2D(padding=((1,0),(0,0)))(reconstruction)
    model = keras.Model([a_input, z_input], reconstruction, name="reconstruction_decoder")
    return model 


def get_decoder_skip_connect(image_0, image_1, hidden_dim, channels=9):
    z_inp = keras.Input((hidden_dim,))
    x = layers.Dense(units=128*int((1/4)*image_0)*(1/4)*image_1)(z_inp)
    x = layers.Reshape(target_shape=(int((1/4)*image_0),int((1/4)*image_1),128))(x)
    x = layers.Conv2DTranspose(128, (5,5), padding="same", strides=(1,1), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (5,5), padding="same", strides=(2,2), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    reconstruction = layers.Conv2DTranspose(channels, (5,5), strides=(2,2), padding="same", activation="tanh")(x)
    if image_0 % 2 != 0:
        reconstruction = layers.ZeroPadding2D(padding=((1,0),(0,0)))(reconstruction)
    model = keras.Model(z_inp, reconstruction, name="reconstruction_decoder")
    return model 



def get_MLP_classifier(hidden_dim, no_classes):
    a = layers.Input((hidden_dim,))
    if no_classes == 2:
        pred = layers.Dense(1, activation='sigmoid')(a)
    else:
        pred = layers.Dense(no_classes, activation='softmax')(a)
    clf = keras.Model(a,pred)
    return clf 


