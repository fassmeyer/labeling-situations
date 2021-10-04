import model_utils
import tensorflow as tf
import keras
import numpy as np
import argparse



# architecture 


class SoccerVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(SoccerVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder  
        
    def encode(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        return z_mean, z_log_var, z
    
    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction 


def build_architecture():
    intermediate_dim = opt.intermediate_dim
    hidden_dim = opt.hidden_dim
    channels = opt.channels

    encoder = model_utils.get_encoder(hidden_dim, intermediate_dim, channels)
    # linear output when with_sigmoid=False
    decoder = model_utils.get_decoder(hidden_dim, intermediate_dim, channels, with_sigmoid=True) 

    return SoccerVAE(encoder, decoder)



def get_model(intermediate_dim=128, hidden_dim=64, channels=9):
    encoder = model_utils.get_encoder(hidden_dim, intermediate_dim, channels)
    # linear output when with_sigmoid=False
    decoder = model_utils.get_decoder(hidden_dim, intermediate_dim, channels, with_sigmoid=True) 

    return SoccerVAE(encoder, decoder)




# training

loss_tracking_metric = keras.metrics.Mean()
kl_tracking_metric = keras.metrics.Mean()
reconstruction_tracking_metric = keras.metrics.Mean()
unweighted_kl_loss_tracking_metric = keras.metrics.Mean()

metrics = [loss_tracking_metric, kl_tracking_metric, reconstruction_tracking_metric, \
    unweighted_kl_loss_tracking_metric]


optimizer = tf.keras.optimizers.Adam(1e-4)


def compute_loss(model,x,kl_weight):
    """Forward Propagation"""

    z_mean, z_log_var, z = model.encode(x)
    reconstruction = model.decode(z)
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2)))
#     reconstruction_loss = tf.keras.losses.MeanSquaredError()(K.flatten(x), K.flatten(reconstruction))
    unweighted_kl_loss = -0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    unweighted_kl_loss = tf.reduce_mean(tf.reduce_sum(unweighted_kl_loss, axis=1))
    kl_loss = kl_weight*unweighted_kl_loss 
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss, unweighted_kl_loss



@tf.function
def train_step(model, x, optimizer, kl_weight):
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



def static_train(model, dataset, epochs, batch_size, channels):

    indices = np.arange(dataset.shape[0])

    for epoch in range(1, epochs+1):
        reset_metrics()
        
        # cost annealing vs standard optimization strategy  
        if epoch < 10:
            kl_weight = (epoch**3)*0.001
        else:
            kl_weight = 1
        # kl_weight = 1
    
        for training_iteration in range(int(dataset.shape[0]//batch_size)): 

            mini_batch = np.zeros((batch_size, 105, 68, channels))
            for i in range(batch_size):
                index = np.random.choice(indices)
                image = dataset[index]
                mini_batch[i] = image

            if np.array(mini_batch).max() > 1:
                mini_batch = clip_values(np.array(mini_batch))

            logs = train_step(model, mini_batch, optimizer, kl_weight)  
        
        
        print('Result at the end of epoch %d:' % (epoch,))
        for key, value in logs.items():
            print('...%s: %.4f' % (key, value))
    



    
def train():
    model = build_architecture()

    epochs = opt.epochs
    batch_size = opt.batch_size
    data = opt.data
    weight_path = opt.weight_path
    channels = opt.channels
    
    # training
    dataset = np.load(data, mmap_mode='r')
    # dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0]).batch(batch_size)

    static_train(model, dataset, epochs, batch_size, channels)
    model.save_weights(weight_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--intermediate-dim', type=int, default=128) 
    parser.add_argument('--hidden-dim', type=int, default=32) 
    parser.add_argument('--channels', type=int, choices=[9, 3], default=9)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)

    # data and weights
    parser.add_argument('--dataset', type=str, required=True, help='Path to game data')
    parser.add_argument('--weights', type=str, required=True, help='Location to save weights')

    opt = parser.parse_args()
    
    train()


