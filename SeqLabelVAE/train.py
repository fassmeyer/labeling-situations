import argparse
import model_utils
import tensorflow as tf
import keras
import numpy as np



def build_architecture():
    feature_dim = opt.feature_dim
    intermediate_dim = opt.intermediate_dim
    hidden_dim = opt.hidden_dim
    pitch_x_axis = opt.pitch_x_axis
    pitch_y_axis = opt.pitch_y_axis
    channels = opt.channels
    timesteps = opt.timesteps
    no_classes = opt.no_classes

    # f: (105, 68, channels) -> (feature_dim,)
    input_feature_extractor = model_utils.get_DCGAN_feature_extractor(feature_dim, pitch_x_axis,\
         pitch_y_axis, channels)
    # f: (timesteps, 105, 68, channels) -> (timesteps, intermediate_dim)
    recurrent_encoder = model_utils.get_recurrent_encoder_timesteps\
    (hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, input_feature_extractor)
    # f: (timesteps, intermediate_dim) -> (timesteps, hidden_dim)
    MLP_a = model_utils.get_timewise_MLP_z(timesteps, intermediate_dim, hidden_dim)

    # f: (timesteps, intermediate_dim + hidden_dim) -> (timesteps, hidden_dim)
    MLP_z = model_utils.get_timewise_MLP_z(timesteps, intermediate_dim, hidden_dim)

    # f: (feature_dim,) -> (105, 68, channels)
    reverse_feature_extractor = model_utils.get_DCGAN_reverse_feature_extractor(feature_dim, pitch_x_axis, \
        pitch_y_axis, channels)

    # f: (timesteps, 2*hidden_dim) -> (timesteps, 105, 68, channels)
    recurrent_decoder = model_utils.get_recurrent_decoder_timesteps_flat_latents\
    (timesteps, hidden_dim, intermediate_dim, feature_dim, reverse_feature_extractor)

    # f: (timesteps, hidden_dim) -> (timesteps, no_classes)
    classifier = model_utils.get_MLP_classifier_timewise(timesteps, hidden_dim, no_classes)

    return SeqLabelVAE(recurrent_encoder, MLP_z, MLP_a, recurrent_decoder, classifier)





class SeqLabelVAE(keras.Model):
    def __init__(self, recurrent_encoder, timewise_MLP_z, timewise_MLP_a, recurrent_decoder, classifier, **kwargs):
        super(SeqLabelVAE, self).__init__(**kwargs)
        self.recurrent_encoder = recurrent_encoder
        self.recurrent_decoder = recurrent_decoder
        self.timewise_MLP_z = timewise_MLP_z
        self.timewise_MLP_a = timewise_MLP_a
        self.classifier = classifier
        
    def encode(self, x):
        h_t = self.recurrent_encoder(x)
        a_mean_t, a_log_sigma_t, a_t = self.timewise_MLP_a(h_t)
        z_mean_t, z_log_sigma_t, z_t = self.timewise_MLP_z([h_t, a_t])
        return a_mean_t, a_log_sigma_t, a_t, z_mean_t, z_log_sigma_t, z_t
    
    def decode(self, a_t, z_t):
        reconstruction = self.recurrent_decoder([a_t,z_t])
        return reconstruction  
    
    def classify(self, a_t):
        y_pred = self.classifier(a_t)
        return y_pred


def get_model(pitch_x_axis=105, pitch_y_axis=68, channels=3, feature_dim=300, intermediate_dim=128, hidden_dim=16, timesteps=20,\
    no_classes = 5):
    # f: (105, 68, channels) -> (feature_dim,)
    input_feature_extractor = model_utils.get_DCGAN_feature_extractor(feature_dim, pitch_x_axis,\
         pitch_y_axis, channels)
    # f: (timesteps, 105, 68, channels) -> (timesteps, intermediate_dim)
    recurrent_encoder = model_utils.get_recurrent_encoder_timesteps\
    (hidden_dim, intermediate_dim, pitch_x_axis, pitch_y_axis, channels, timesteps, input_feature_extractor)
    # f: (timesteps, intermediate_dim) -> (timesteps, hidden_dim)
    MLP_a = model_utils.get_timewise_MLP_z(timesteps, intermediate_dim, hidden_dim)

    # f: (timesteps, intermediate_dim + hidden_dim) -> (timesteps, hidden_dim)
    MLP_z = sequential_mdoels.get_timewise_MLP_z_1(timesteps, intermediate_dim, hidden_dim)

    # f: (feature_dim,) -> (105, 68, channels)
    reverse_feature_extractor = model_utils.get_DCGAN_reverse_feature_extractor(feature_dim, pitch_x_axis, \
        pitch_y_axis, channels)

    # f: (timesteps, 2*hidden_dim) -> (timesteps, 105, 68, channels)
    recurrent_decoder = model_utils.get_recurrent_decoder_timesteps_flat_latents\
    (timesteps, hidden_dim, intermediate_dim, feature_dim, reverse_feature_extractor)

    # f: (timesteps, hidden_dim) -> (timesteps, no_classes)
    classifier = model_utils.get_MLP_classifier_timewise(timesteps, hidden_dim, no_classes)

    return SeqLabelVAE(recurrent_encoder, MLP_z, MLP_a, recurrent_decoder, classifier)


# Training
loss_tracking_metric = keras.metrics.Mean()
kl_tracking_metric_z = keras.metrics.Mean()
kl_tracking_metric_a = keras.metrics.Mean()
reconstruction_tracking_metric = keras.metrics.Mean()
classification_loss_tracking_metric = keras.metrics.Mean()
kl_tracking_metric = keras.metrics.Mean()
optimizer = tf.keras.optimizers.Adam(lr=0.0001) # chosen according to Ha & Eck (2017)

metrics = [loss_tracking_metric, kl_tracking_metric_z, kl_tracking_metric_a, reconstruction_tracking_metric,\
    classification_loss_tracking_metric, kl_tracking_metric]



def compute_loss(model, x_labeled, y, x_unlabeled, labeled_batch_size, unlabeled_batch_size, \
    kl_weight, timesteps, alpha=0.1):
    """Forward Propagation"""

    # x_unlabeled -> shape = (batch_size, timesteps + 1, 105, 68, channels)
    x = tf.keras.backend.concatenate([tf.cast(x_unlabeled, "float32"),\
        tf.cast(x_labeled, "float32")], axis=0)
    a_mean_t, a_log_var_t, a_t, z_mean_t, z_log_var_t, z_t = model.encode(x)

    # compute reconstruction loss only on unlabeled part
    reconstruction = model.decode(a_t[:unlabeled_batch_size], z_t[:unlabeled_batch_size])

    # Evaluate -log p(x|z,a) for the unlabeled mini-batch; z_t and a_t participate to x_t+1
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(x_unlabeled, \
        reconstruction), axis=(1, 2, 3)))

    # Evaluate KL
    kl_loss_z = -0.5*(1 + z_log_var_t - tf.square(z_mean_t) - tf.exp(z_log_var_t))
    kl_loss_z = tf.reduce_mean(tf.reduce_sum(kl_loss_z, axis=(1,2)))
    kl_loss_a = -0.5*(1 + a_log_var_t[:unlabeled_batch_size] - \
                                 tf.square(a_mean_t[:unlabeled_batch_size]) - tf.exp(a_log_var_t[:unlabeled_batch_size]))
    kl_loss_a = tf.reduce_mean(tf.reduce_sum(kl_loss_a, axis=(1,2)))

    kl_min = 2
    # kl_weight(epoch)*L_KL
    kl_loss = kl_weight*(keras.backend.maximum(kl_loss_z, kl_min) + keras.backend.maximum(kl_loss_a, kl_min)) 

    # Evaluate -log q(y|a) for the labeled part of the mini-batch
    a_classifier_inp = a_t[-labeled_batch_size:]
    y_pred = model.classify(a_classifier_inp)
    classification_loss_un = tf.reduce_mean(tf.reduce_sum(keras.losses.SparseCategoricalCrossentropy\
        (reduction=tf.keras.losses.Reduction.NONE)(tf.reshape(tf.repeat(y, timesteps), \
            (labeled_batch_size,timesteps)),y_pred), axis=-1))
    classification_loss = alpha*((unlabeled_batch_size+labeled_batch_size)/labeled_batch_size)*classification_loss_un

    # L = L_R + kl_weight*(L_KLz + L_KLy) + alpha*L_CL
    total_loss = reconstruction_loss + kl_loss + classification_loss

    return total_loss, reconstruction_loss, kl_loss_z, kl_loss_a, classification_loss_un, kl_loss


@tf.function
def train_step(model, x_labeled, y, x_unlabeled, optimizer, labeled_batch_size, unlabeled_batch_size, 
     timesteps, kl_weight):
    with tf.GradientTape() as tape:
        loss, reconstruction_loss, kl_loss_z, kl_loss_a, classification_loss, kl_loss = compute_loss\
        (model, x_labeled, y, x_unlabeled, labeled_batch_size, unlabeled_batch_size, 
            kl_weight, timesteps)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
    logs = {}
    loss_tracking_metric.update_state(loss)
    kl_tracking_metric_z.update_state(kl_loss_z)
    kl_tracking_metric_a.update_state(kl_loss_a)
    reconstruction_tracking_metric.update_state(reconstruction_loss)
    classification_loss_tracking_metric.update_state(classification_loss)
    kl_tracking_metric.update_state(kl_loss)
    
    logs['loss'] = loss_tracking_metric.result()
    logs['kl_loss_z (unweighted)'] = kl_tracking_metric_z.result()
    logs['kl_loss_a (unweighted)'] = kl_tracking_metric_a.result()
    logs['reconstruction_loss'] = reconstruction_tracking_metric.result()
    logs['classification_loss (unweighted)'] = classification_loss_tracking_metric.result()
    logs['kl_loss (weighted)'] = kl_tracking_metric.result()
    return logs 



def SeqLabelVAE_train(model, unlabeled_dataset, labeled_dataset, y, epochs, timesteps, labeled_batch_size,\
    unlabeled_batch_size, channels, cost_annealing=True):
    """Dataset: (no_obs, 105, 68, 3); Generates 'on-the-fly' sequences for specified timesteps
    Labeled Dataset: (no_obs, timesteps, 105, 68, 3) with y encoding the associated labels for the sequences"""

    unlabeled_start_indices = np.arange(unlabeled_dataset.shape[0]-timesteps-1)
    labeled_start_indices = np.arange(labeled_dataset.shape[0])

    for epoch in range(1, epochs+1):
        reset_metrics()

        if const_annealing:
            # epoch-wise annealing schedule
            if epoch < 10:
                kl_weight = (epoch**3)*0.001
            else:
                kl_weight = 1
        else:
            kl_weight = 1

        for training_iteration in range(int(unlabeled_dataset.shape[0]//unlabeled_batch_size - timesteps)):
            
            x_unlabeled  = np.zeros((unlabeled_batch_size, timesteps, 105, 68, channels)) 
            for i in range(unlabeled_batch_size):
                start_index = np.random.choice(unlabeled_start_indices)
                images = unlabeled_dataset[start_index:start_index + timesteps]
                x_unlabeled[i] = images

            indices = np.random.choice(labeled_start_indices, size=(labeled_batch_size))
            x_labeled  = labeled_dataset[indices]
            y_mini_batch = y[indices]
                
            # # Clip values if the image batch has target values larger than 1
            # if x_unlabeled.max() > 1:
            #     x_unlabeled = clip_values(x_unlabeled)

            # # Clip values if the image batch has target values larger than 1
            # if x_labeled.max() > 1:
            #     x_labeled = clip_values(x_labeled)

                
            logs = train_step(model, x_labeled, y_mini_batch, x_unlabeled, optimizer,\
                 labeled_batch_size, unlabeled_batch_size, timesteps, kl_weight = kl_weight)

                    
        print('Result at the end of epoch %d:' % (epoch,))
        for key, value in logs.items():
            print('...%s: %.4f' % (key, value))





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


def train():
    # model architecture
    model = build_architecture()

    # training hyperparameters
    epochs = opt.epochs
    timesteps = opt.timesteps
    unlabeled_batch_size = opt.unlabeled_batch_size
    labeled_batch_size = opt.labeled_batch_size
    weight_path = opt.weight_path
    channels = opt.channels
    cost_annealing = opt.cost_annealing

    # data
    labels = opt.labels
    labeled_images = opt.labeled_images
    unlabeled_images = opt.unlabeled_images
    

    # training dataset
    y = np.load(labels, mmap_mode='r')

    labeled_dataset = np.load(labeled_images, mmap_mode='r')
    unlabeled_dataset = np.load(unlabeled_images, mmap_mode='r')
    
    # model training
    SeqLabelVAE_train(model, unlabeled_dataset, labeled_dataset, y, epochs, timesteps, labeled_batch_size,\
         unlabeled_batch_size, channels, cost_annealing)
    model.save_weights(weight_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--feature-dim', type=int, default=300) 
    parser.add_argument('--intermediate-dim', type=int, default=128) 
    parser.add_argument('--hidden-dim', type=int, default=8) 
    parser.add_argument('--pitch-x-axis', type=int, default=105)
    parser.add_argument('--pitch-y-axis', type=int, default=68)
    parser.add_argument('--channels', type=int, choices=[3,9], default=9)
    parser.add_argument('--timesteps', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--unlabeled-batch-size', type=int, default=64)
    parser.add_argument('--labeled-batch-size', type=int, default=4)
    parser.add_argument('--no-classes', type=int, default=5)
    parser.add_argument('--cost-annealing', type=bool, default=False)

    # data + weights
    parser.add_argument('--labeled-sequences', type=str, required=True,\
     help='Path to sequences containing the action of interest. Shape = (n_obs, seq_timesteps, 105, 68, 9)') 
    parser.add_argument('--labels', type=str, required=True, help='Path to associated labels') 
    parser.add_argument('--unlabeled-frames', type=str, required=True,\
     help='Path to (static) game data. Shape = (game_timesteps, 105, 68, 9)') 
    parser.add_argument('--weights', type=str, required=True, help='Location to save weights')
    
    opt = parser.parse_args()

    model = build_architecture()
    train()
