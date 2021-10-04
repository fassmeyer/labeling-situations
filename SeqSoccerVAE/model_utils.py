import tensorflow as tf
import keras 
from keras import layers
from keras import backend as K
import numpy as np
from keras import losses
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder



#------------------------------------------Sequential M1-------------------------------------------------
class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch_size = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
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



def get_recurrent_encoder(hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, feature_extractor):
    game_sequence = layers.Input(shape=(timesteps, pitch_x_axis, pitch_y_axis, channels))
    feature_sequence = layers.TimeDistributed(feature_extractor)(game_sequence)
    h = layers.LSTM(intermediate_dim)(feature_sequence)
    z_mean = layers.Dense(hidden_dim)(h)
    z_log_sigma = layers.Dense(hidden_dim)(h)
    z = Sampling()([z_mean, z_log_sigma])
    encoder = keras.Model(game_sequence, [z_mean, z_log_sigma, z], name="encoder")
    return encoder



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




#----------------------------------------Composite Model-------------------------------------------------------------------------
def get_input_feature_extractor(x_size, y_size, channels, feature_dim):
    tracking_frame = layers.Input(shape=(x_size, y_size, channels))
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(tracking_frame) # -> (x_coordinate, y_coordinate, channels)
    x = layers.Conv2D(64, (3,3), padding="same", activation = "relu", strides=(2,2))(x) 
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    feature_vector = layers.Dense(feature_dim, activation="relu")(x)
    feature_dim = K.int_shape(feature_vector)[1]
    input_feature_extractor = keras.Model(tracking_frame, feature_vector, name="input_feature_extractor")
    return input_feature_extractor
    

def get_encoder_LSTM(reconstr_timesteps, x_size, y_size, channels, intermediate_dim, hidden_dim, input_feature_extractor):
    game_sequence = layers.Input(shape=(reconstr_timesteps, x_size, y_size, channels))
    feature_sequence = layers.TimeDistributed(input_feature_extractor)(game_sequence)
    h = layers.LSTM(intermediate_dim)(feature_sequence)
    z = layers.Dense(hidden_dim)(h)
    encoder = keras.Model(game_sequence, z, name="encoder_lstm")
    return encoder


def get_output_feature_extractor(channels, feature_dim):
    f_vector = layers.Input(shape=(feature_dim,))
    x = layers.Dense(units = 115328)(f_vector)
    x = layers.Reshape(target_shape=(53, 34, 64))(x)
    x = layers.Conv2DTranspose(64, (3,3), padding="same", activation = "relu", strides=(2,2))(x)
    x = layers.Cropping2D(cropping=((1,0),(0,0)))(x)
    t_frame = layers.Conv2D(channels, (3,3), padding="same", activation="relu")(x)
    output_feature_extractor = keras.Model(f_vector, t_frame, name="output_feature_extractor")
    return output_feature_extractor


def get_future_output_feature_extractor(feature_dim, channels):
    f_vector_future = layers.Input(shape=(feature_dim,))
    x = layers.Dense(units = 115328)(f_vector_future)
    x = layers.Reshape(target_shape=(53, 34, 64))(x)
    x = layers.Conv2DTranspose(64, (3,3), padding="same", activation = "relu", strides=(2,2))(x)
    x = layers.Cropping2D(cropping=((1,0),(0,0)))(x)
    t_frame_future = layers.Conv2D(channels, (3,3), padding="same", activation="relu")(x)
    output_feature_extractor_future = keras.Model(f_vector_future, t_frame_future, name="output_feature_extractor_future")
    return output_feature_extractor_future


def get_reconstruction_decoder(hidden_dim, reconstr_timesteps, intermediate_dim, feature_dim, output_feature_extractor):
    hidden_representation = layers.Input(shape=(hidden_dim,))
    repeated_z = layers.RepeatVector(reconstr_timesteps)(hidden_representation)
    decoder_h = layers.LSTM(intermediate_dim, return_sequences=True)(repeated_z)
    decoder_mean = layers.LSTM(feature_dim, return_sequences=True)(decoder_h)
    x_decoded_mean_sequence = layers.TimeDistributed(output_feature_extractor)(decoder_mean)
    reconstruction_decoder = keras.Model(hidden_representation, x_decoded_mean_sequence, name="Input_Reconsstruction")
    return reconstruction_decoder


def get_future_decoder(future_timesteps, hidden_representation, intermediate_dim, feature_dim, output_feature_extractor_future):
    repeated_z_future = layers.RepeatVector(future_timesteps)(hidden_representation)
    future_decoder_h = layers.LSTM(intermediate_dim, return_sequences=True)(repeated_z_future)
    future_decoder_mean = layers.LSTM(feature_dim, return_sequences=True)(future_decoder_h)
    x_decoded_mean_sequence_future = layers.TimeDistributed(output_feature_extractor_future)(future_decoder_mean)
    future_decoder = keras.Model(hidden_representation, x_decoded_mean_sequence_future, name="Future_Prediction")
    return future_decoder



#------------------------------------------Sequential VAE------------------------------------------------------------------------------
def get_frame_feature_extractor(image_0, image_1, channels):
    inp = layers.Input((image_0, image_1, channels))
    x = layers.Conv2D(32, (5,5), strides=(2,2), padding='same')(inp)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (5,5), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    out = layers.Dense(units = 100)(x)
    frame_feature_extractor = keras.Model(inp, out, name="frame_feature_extractor")
    return frame_feature_extractor


def get_encoder_and_sampling(image_0, image_1, channels, timesteps, hidden_dim, feature_extractor, units=1568):
    model_input = layers.Input((timesteps, image_0, image_1, channels))
    feature_sequence = layers.TimeDistributed(feature_extractor)(model_input)
    h_enc = layers.LSTM(units)(feature_sequence)
    z_mean = layers.Dense(hidden_dim)(h_enc)
    z_log_sigma = layers.Dense(hidden_dim)(h_enc)
    z_t = Sampling()([z_mean, z_log_sigma])
    model = keras.Model(model_input, [z_mean, z_log_sigma, z_t], name='representation_extractor')
    return model


def get_reconstruction_model(image_0, image_1, hidden_dim, channels):
    z_input = keras.Input((hidden_dim,))
    x = layers.Dense(units=128*int((1/4)*image_0)*(1/4)*image_1)(z_input)
    x = layers.Reshape(target_shape=(int((1/4)*image_0),int((1/4)*image_1),128))(x)
    x = layers.Conv2DTranspose(64, (5,5), padding="same", strides=(1,1), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (5,5), padding="same", strides=(2,2), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    reconstruction = layers.Conv2DTranspose(channels, (5,5), strides=(2,2), padding="same", activation="tanh")(x)
    if image_0 % 2 != 0:
        reconstruction = layers.ZeroPadding2D(padding=((1,0),(0,0)))(reconstruction)
    model = keras.Model(z_input, reconstruction, name="reconstruction_decoder")
    return model  