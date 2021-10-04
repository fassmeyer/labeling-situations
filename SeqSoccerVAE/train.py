import argparse
import model_utils
import tensorflow as tf
import keras
import numpy as np



class SeqSoccerVAE(keras.Model):
    def __init__(self, recurrent_encoder, recurrent_decoder, **kwargs):
        super(SeqSoccerVAE, self).__init__(**kwargs)
        self.recurrent_encoder = recurrent_encoder
        self.recurrent_decoder = recurrent_decoder  
        
    def encode(self, x):
        z_mean, z_log_var, z = self.recurrent_encoder(x)
        return z_mean, z_log_var, z
    
    def decode(self, z):
        reconstruction = self.recurrent_decoder(z)
        return reconstruction    


def build_architecture():
    feature_dim = opt.feature_dim
    intermediate_dim = opt.intermediate_dim
    hidden_dim = opt.hidden_dim
    pitch_x_axis = opt.pitch_x_axis
    pitch_y_axis = opt.pitch_y_axis
    channels = opt.channels
    timesteps = opt.timesteps

    input_feature_extractor = model_utils.get_feature_extractor(feature_dim, pitch_x_axis, pitch_y_axis, channels)
    recurrent_encoder = model_utils.get_recurrent_encoder\
    (hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, input_feature_extractor)
    reverse_feature_extractor = model_utils.get_reverse_feature_extractor(feature_dim, pitch_x_axis, pitch_y_axis, channels)
    recurrent_decoder = model_utils.get_recurrent_decoder\
    (feature_dim, hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, reverse_feature_extractor)

    return SeqSoccerVAE(recurrent_encoder, recurrent_decoder)
    

loss_tracking_metric = keras.metrics.Mean()
kl_tracking_metric = keras.metrics.Mean()
reconstruction_tracking_metric = keras.metrics.Mean()
unweighted_kl_loss_tracking_metric = keras.metrics.Mean()
optimizer = tf.keras.optimizers.Adam(lr=0.0001)

metrics = [loss_tracking_metric, kl_tracking_metric, reconstruction_tracking_metric, unweighted_kl_loss_tracking_metric]


def compute_loss(model, x, kl_weight):
    """Forward Propagation"""
    z_mean, z_log_var, z = model.encode(x)
    # reconstructing the sequence of frames
    reconstruction = model.decode(z)
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2, 3)))
    unweighted_kl_loss = -0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    unweighted_kl_loss = tf.reduce_mean(tf.reduce_sum(unweighted_kl_loss, axis=1))
    # Loss_train = L_R + kl_weight(epoch)*L_KL
    kl_loss = kl_weight*unweighted_kl_loss 
    # KL_min = 0.5 # free-bits
    # if np.array(unweighted_kl_loss) >= KL_min:
    total_loss = reconstruction_loss + kl_loss 
    # else: #if kl_loss is below a certain threshold -> put focus on minimizing the reconstruction loss
    #     total_loss = reconstruction_loss + kl_weight*KL_min
    return total_loss, reconstruction_loss, kl_loss, unweighted_kl_loss


@tf.function
def train_step(model, x, optimizer, kl_weight):
    """Update the weights using SGD optimizer"""
    with tf.GradientTape() as tape:
        loss, reconstruction_loss, kl_loss, unweighted_kl_loss = compute_loss(model, x, kl_weight)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    logs = {}
    loss_tracking_metric.update_state(loss)
    kl_tracking_metric.update_state(kl_loss)
    reconstruction_tracking_metric.update_state(reconstruction_loss)
    unweighted_kl_loss_tracking_metric.update_state(unweighted_kl_loss)
    
    logs['loss'] = loss_tracking_metric.result()
    logs['kl_loss'] = kl_tracking_metric.result()
    logs['reconstruction_loss'] = reconstruction_tracking_metric.result()
    logs['unweighted_kl_loss'] = unweighted_kl_loss_tracking_metric.result()
    return logs


def clip_values(image_batch):
    """Goes over each element of the image batch clips it to 1 if it exceeds the value 1"""
    iterator = image_batch.flatten()
    for i, element in enumerate(iterator):
        if element <= 1:
            continue
        else:
            iterator[i] = 1
    return np.reshape(iterator, image_batch.shape)


def reset_metrics():
    for metric in metrics:
        metric.reset_states()


def seqsoccervae_train(dataset, model, epochs, timesteps, batch_size, channels):
    """Dataset: (no_obs, 105, 68, 3); Generates 'on-the-fly' sequences for specified timesteps"""

    weight_path = opt.weight_path
    
    all_start_indices = np.arange(dataset.shape[0]-(timesteps-1))

    for epoch in range(1, epochs+1):
        reset_metrics()

        # epoch-wise annealing schedule
        if epoch < 10:
            kl_weight = (epoch**3)*0.001
        else:
            kl_weight = 1

        for training_iteration in range(int(dataset.shape[0]//batch_size - timesteps)):
                
            mini_batch  = np.zeros((batch_size, timesteps, 105, 68, channels)) 
            for i in range(batch_size):
                start_index = np.random.choice(all_start_indices)
                images = dataset[start_index:start_index + timesteps]
                mini_batch[i] = images
                
            # # Clip values if the image batch has target values larger than 1
            # if mini_batch.max() > 1:
            #     mini_batch = clip_values(mini_batch)
                
            # Apply a full training iteration
            logs =  train_step(model, mini_batch, optimizer, kl_weight)  

        # ensure fault tolerance 
        model.save_weights(weight_path)
                    
        print('Result at the end of epoch %d:' % (epoch,))
        for key, value in logs.items():
            print('...%s: %.4f' % (key, value))



def get_model(pitch_x_axis=105, pitch_y_axis=68, channels=3, feature_dim=300, intermediate_dim=512, hidden_dim=128, timesteps=20):
    input_feature_extractor = model_utils.get_feature_extractor(feature_dim, pitch_x_axis, pitch_y_axis, channels)
    recurrent_encoder = model_utils.get_recurrent_encoder\
    (hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, input_feature_extractor)
    reverse_feature_extractor = model_utils.get_reverse_feature_extractor(feature_dim, pitch_x_axis, pitch_y_axis, channels)
    recurrent_decoder = model_utils.get_recurrent_decoder\
    (feature_dim, hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, reverse_feature_extractor)

    return Sequential_VAE(recurrent_encoder, recurrent_decoder)



def train(model):

    epochs = opt.epochs
    timesteps = opt.timesteps
    batch_size = opt.batch_size
    data = opt.data
    weight_path = opt.weight_path
    channels = opt.channels
    

    # training
    data = np.load(data, mmap_mode='r')
    seqsoccervae_train(data, model, epochs, timesteps, batch_size, channels)
    model.save_weights(weight_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-dim', type=int, default=300) 
    parser.add_argument('--intermediate-dim', type=int, default=512) 
    parser.add_argument('--hidden-dim', type=int, default=128) 
    parser.add_argument('--pitch-x-axis', type=int, default=105)
    parser.add_argument('--pitch-y-axis', type=int, default=68)
    parser.add_argument('--channels', type=int, default=9)
    parser.add_argument('--timesteps', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)

    # data + weights
    parser.add_argument('--dataset', type=str, required=True, help='Game data')
    parser.add_argument('--weights', type=str, required=True, help='Location to save weights')

    opt = parser.parse_args()

    model = build_architecture()
    train(model)




