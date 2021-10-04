import tensorflow as tf
import keras 
from keras import layers
from keras import backend as K
import numpy as np
from keras import losses
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder



class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch_size = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Sampling_timestep(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch_size = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[2]
    timesteps = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch_size, timesteps, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_feature_extractor(feature_dim, pitch_x_axis, pitch_y_axis, channels):
    tracking_frame = layers.Input(shape=(pitch_x_axis, pitch_y_axis, channels))
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(tracking_frame) # -> (x_coordinate, y_coordinate, channels)
    x = layers.Conv2D(64, (3,3), padding="same", activation = "relu", strides=(2,2))(x) 
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    feature_vector = layers.Dense(feature_dim, activation="relu")(x)
    input_feature_extractor = keras.Model(tracking_frame, feature_vector, name="input_feature_extractor")
    return input_feature_extractor



def get_DCGAN_feature_extractor(feature_dim, image_0, image_1, channels):
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


def get_timewise_feature_extractor(feature_dim, image_0, image_1, channels, timesteps, feature_extractor):
    game_sequence = layers.Input((timesteps, image_0, image_1, channels))
    feature_sequence = layers.TimeDistributed(feature_extractor)(game_sequence)
    model = keras.Model(game_sequence, feature_sequence)
    return model

def get_recurrent_encoder_timesteps(hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels,\
                                    timesteps, feature_extractor):
    game_sequence = layers.Input(shape=(timesteps, pitch_x_axis, pitch_y_axis, channels))
    feature_sequence = layers.TimeDistributed(feature_extractor)(game_sequence)
    h = layers.LSTM(intermediate_dim, return_sequences=True)(feature_sequence)
#     z_mean = layers.Dense(hidden_dim)(h)
#     z_log_sigma = layers.Dense(hidden_dim)(h)
#     z = Sampling()([z_mean, z_log_sigma])
    encoder = keras.Model(game_sequence, h, name="encoder")
    return encoder


def get_recurrent_encoder(hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, feature_extractor):
    game_sequence = layers.Input(shape=(timesteps, pitch_x_axis, pitch_y_axis, channels))
    feature_sequence = layers.TimeDistributed(feature_extractor)(game_sequence)
    h = layers.LSTM(intermediate_dim)(feature_sequence)
    z_mean = layers.Dense(hidden_dim)(h)
    z_log_sigma = layers.Dense(hidden_dim)(h)
    z = Sampling()([z_mean, z_log_sigma])
    encoder = keras.Model(game_sequence, [z_mean, z_log_sigma, z], name="encoder")
    return encoder

def get_timewise_MLP_a(timesteps, intermediate_dim, hidden_dim):
    inp = layers.Input((timesteps, intermediate_dim))
    z_mean_t = layers.TimeDistributed(layers.Dense(hidden_dim))(inp)
    z_log_sigma_t = layers.TimeDistributed(layers.Dense(hidden_dim))(inp)
    z_t = Sampling_timestep()([z_mean_t, z_log_sigma_t])
    model = keras.Model(inp, [z_mean_t, z_log_sigma_t, z_t])
    return model

def get_timewise_MLP_z(timesteps, intermediate_dim, hidden_dim):
    feature_inp = layers.Input((timesteps, intermediate_dim))
    hidden_inp = layers.Input((timesteps, hidden_dim))
    inp = layers.Concatenate()([feature_inp, hidden_inp])
    z_mean_t = layers.TimeDistributed(layers.Dense(hidden_dim))(inp)
    z_log_sigma_t = layers.TimeDistributed(layers.Dense(hidden_dim))(inp)
    z_t = Sampling_timestep()([z_mean_t, z_log_sigma_t])
    model = keras.Model([feature_inp, hidden_inp], [z_mean_t, z_log_sigma_t, z_t])
    return model


def get_forward_encoder(intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, feature_extractor):
    game_sequence = layers.Input(shape=(timesteps, pitch_x_axis, pitch_y_axis, channels))
    feature_sequence = layers.TimeDistributed(feature_extractor)(game_sequence)
    h = layers.LSTM(intermediate_dim)(feature_sequence)
    model = keras.Model(game_sequence, h)
    return model


def get_backward_encoder(intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, feature_extractor):
    game_sequence_reversed = layers.Input(shape=(timesteps, pitch_x_axis, pitch_y_axis, channels))
    feature_sequence = layers.TimeDistributed(feature_extractor)(game_sequence_reversed)
    h = layers.LSTM(intermediate_dim)(feature_sequence)
    model = keras.Model(game_sequence_reversed, h)
    return model


def get_MLP_z(hidden_dim, intermediate_dim):
    h_forward = layers.Input((intermediate_dim,))
    h_backward = layers.Input((intermediate_dim,))
    h = layers.Concatenate()([h_forward, h_backward])
    z_mean = layers.Dense(hidden_dim)(h)
    z_log_sigma = layers.Dense(hidden_dim)(h)
    z = Sampling()([z_mean, z_log_sigma])
    model = keras.Model([h_forward, h_backward], [z_mean, z_log_sigma, z])
    return model


def get_MLP_a(hidden_dim, intermediate_dim):
    h_forward = layers.Input((intermediate_dim,))
    h_backward = layers.Input((intermediate_dim,))
    h = layers.Concatenate()([h_forward, h_backward])
    a_mean = layers.Dense(hidden_dim)(h)
    a_log_sigma = layers.Dense(hidden_dim)(h)
    a = Sampling()([a_mean, a_log_sigma])
    model = keras.Model([h_forward, h_backward], [a_mean, a_log_sigma, a])
    return model


def get_MLP_z_LabelVAE(hidden_dim, intermediate_dim):
    h_forward = layers.Input((intermediate_dim,))
    h_backward = layers.Input((intermediate_dim,))
    a = layers.Input((hidden_dim,))
    inp = layers.Concatenate()([h_forward, h_backward, a])
    z_mean = layers.Dense(hidden_dim)(inp)
    z_log_sigma = layers.Dense(hidden_dim)(inp)
    z = Sampling()([z_mean, z_log_sigma])
    model = keras.Model([h_forward, h_backward, a], [z_mean, z_log_sigma, z])
    return model


def get_MLP_classifier(hidden_dim, no_classes):
    a = layers.Input((hidden_dim,))
    if no_classes == 2:
        pred = layers.Dense(1, activation='sigmoid')(a)
    else:
        pred = layers.Dense(no_classes, activation='softmax')(a)
    clf = keras.Model(a,pred)
    return clf 


def get_MLP_classifier_timewise(timesteps, hidden_dim, no_classes):
    a = layers.Input((timesteps, hidden_dim))
    if no_classes == 2:
        pred = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(a)
    else:
        pred = layers.TimeDistributed(layers.Dense(no_classes, activation='softmax'))(a)
    clf = keras.Model(a,pred)
    return clf 



def get_reverse_feature_extractor(feature_dim, pitch_x_axis, pitch_y_axis, channels):
    f_vector = layers.Input(shape=(feature_dim,))
    x = layers.Dense(units = 115328)(f_vector)
    x = layers.Reshape(target_shape=(53, 34, 64))(x)
    x = layers.Conv2DTranspose(64, (3,3), padding="same", activation = "relu", strides=(2,2))(x)
    x = layers.Cropping2D(cropping=((1,0),(0,0)))(x)
    t_frame = layers.Conv2D(channels, (3,3), padding="same", activation="sigmoid")(x)
    output_feature_extractor = keras.Model(f_vector, t_frame, name="output_feature_extractor")
    return output_feature_extractor


def get_DCGAN_reverse_feature_extractor(feature_dim, image_0, image_1, channels):
    z_inp = keras.Input((feature_dim,))
    x = layers.Dense(units=128*int((1/4)*image_0)*(1/4)*image_1)(z_inp)
    x = layers.Reshape(target_shape=(int((1/4)*image_0),int((1/4)*image_1),128))(x)
    x = layers.Conv2DTranspose(128, (5,5), padding="same", strides=(1,1), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (5,5), padding="same", strides=(2,2), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    reconstruction = layers.Conv2DTranspose(channels, (5,5), strides=(2,2), padding="same", activation="tanh")(x)
    reconstruction = layers.Conv2DTranspose(channels, (5,5), strides=(1,1), padding = "same", activation="sigmoid")(reconstruction)
    if image_0 % 2 != 0:
        reconstruction = layers.ZeroPadding2D(padding=((1,0),(0,0)))(reconstruction)
    model = keras.Model(z_inp, reconstruction, name="reconstruction_decoder")
    return model 



def get_recurrent_decoder(feature_dim, hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, reverse_feature_extractor):
    hidden_representation = layers.Input(shape=(hidden_dim,))
    repeated_z = layers.RepeatVector(timesteps)(hidden_representation)
    decoder_h = layers.LSTM(intermediate_dim, return_sequences=True)(repeated_z)
    decoder_mean = layers.LSTM(feature_dim, return_sequences=True)(decoder_h)
    x_decoded_mean_sequence = layers.TimeDistributed(reverse_feature_extractor)(decoder_mean)
    decoder = keras.Model(hidden_representation, x_decoded_mean_sequence)
    return decoder

def get_recurrent_decoder_LabelVAE(feature_dim, hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, reverse_feature_extractor):
    a = layers.Input(shape=(hidden_dim,))
    z = layers.Input(shape=(hidden_dim,))
    inp = layers.Concatenate()([a,z])
    repeated_hidden = layers.RepeatVector(timesteps)(inp)
    decoder_h = layers.LSTM(intermediate_dim, return_sequences=True)(repeated_hidden)
    decoder_mean = layers.LSTM(feature_dim, return_sequences=True)(decoder_h)
    x_decoded_mean_sequence = layers.TimeDistributed(reverse_feature_extractor)(decoder_mean)
    decoder = keras.Model([a,z], x_decoded_mean_sequence)
    return decoder


def get_recurrent_decoder_timesteps(timesteps, hidden_dim, intermediate_dim, feature_dim, reverse_feature_extractor):
    inp = layers.Input(shape=(timesteps, hidden_dim))
    decoder_h = layers.LSTM(intermediate_dim, return_sequences=True)(inp)
    decoder_mean = layers.LSTM(feature_dim, return_sequences=True)(decoder_h)
    x_decoded_mean_sequence = layers.TimeDistributed(reverse_feature_extractor)(decoder_mean)
    decoder = keras.Model(inp, x_decoded_mean_sequence)
    return decoder


def get_recurrent_decoder_timesteps_flat_latents(timesteps, hidden_dim, intermediate_dim, feature_dim, reverse_feature_extractor):
    inp_a = layers.Input((timesteps, hidden_dim))
    inp_z = layers.Input((timesteps, hidden_dim))
    inp = layers.Concatenate()([inp_a, inp_z])
    decoder_h = layers.LSTM(intermediate_dim, return_sequences=True)(inp)
    decoder_mean = layers.LSTM(feature_dim, return_sequences=True)(decoder_h)
    x_decoded_mean_sequence = layers.TimeDistributed(reverse_feature_extractor)(decoder_mean)
    decoder = keras.Model([inp_a, inp_z], x_decoded_mean_sequence)
    return decoder

