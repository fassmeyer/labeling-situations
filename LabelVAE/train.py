import numpy as np
import tensorflow as tf
import keras
import static_utils
import argparse



# architecture

class LabelVAE(keras.Model):
    def __init__(self, feature_extractor, MLP_a, MLP_z, decoder, classifier, **kwargs):
        super(LabelVAE, self).__init__(**kwargs)
        self.feature_extractor = feature_extractor
        self.MLP_z = MLP_z
        self.MLP_a = MLP_a
        self.decoder = decoder
        self.classifier = classifier
    
    def encode(self, x):
        f = self.feature_extractor(x)
        a_mean, a_log_var, a = self.MLP_a(f)
        z_mean, z_log_var, z = self.MLP_z([f, a])
        return a_mean, a_log_var, a, z_mean, z_log_var, z
    
    def decode(self, a, z):
        reconstruction = self.decoder([a, z])
        return reconstruction
    
    def classify(self,a):
        label_pred = self.classifier(a)
        return label_pred




def build_architecture():
    image_0 = opt.image_0
    image_1 = opt.image_1
    channels = opt.channels
    feature_dim = opt.feature_dim
    hidden_dim = opt.hidden_dim
    no_classes = opt.no_classes


    feature_extractor = static_utils.get_feature_extractor(image_0, image_1, channels, feature_dim)
    MLP_a = static_utils.get_MLP_a(feature_dim, hidden_dim)
    MLP_z = static_utils.get_MLP_z(feature_dim, hidden_dim)
    decoder = static_utils.get_decoder(image_0, image_1, hidden_dim, channels)
    classifier = static_utils.get_MLP_classifier(hidden_dim, no_classes)
    return LabelVAE(feature_extractor, MLP_a, MLP_z, decoder, classifier)


def get_model(image_0 = 105, image_1 = 68, channels = 3, feature_dim = 300, hidden_dim = 16, no_classes = 4):
    feature_extractor = static_utils.get_feature_extractor(image_0, image_1, channels, feature_dim)
    MLP_a = static_utils.get_MLP_a(feature_dim, hidden_dim)
    MLP_z = static_utils.get_MLP_z(feature_dim, hidden_dim)
    decoder = static_utils.get_decoder(image_0, image_1, hidden_dim, channels)
    classifier = static_utils.get_MLP_classifier(hidden_dim, no_classes)
    return LabelVAE(feature_extractor, MLP_a, MLP_z, decoder, classifier)



# training

loss_tracking_metric = keras.metrics.Mean()
kl_tracking_metric_z = keras.metrics.Mean()
kl_tracking_metric_a = keras.metrics.Mean()
reconstruction_tracking_metric = keras.metrics.Mean()
classification_loss_tracking_metric = keras.metrics.Mean()
kl_tracking_metric = keras.metrics.Mean()

metrics = [loss_tracking_metric, kl_tracking_metric_z, kl_tracking_metric_a, reconstruction_tracking_metric,\
    classification_loss_tracking_metric, kl_tracking_metric]


def reset_metrics():
    for metric in metrics:
        metric.reset_states()

optimizer = tf.keras.optimizers.Adam(lr=0.0001) # chosen according to Ha & Eck (2017)

def compute_loss(model, x_labeled, y, x_unlabeled, labeled_batch_size, unlabeled_batch_size, kl_weight=1, alpha=0.1):
    """Forward Propagation"""
    

    # Step 1.- 4.:
    x = tf.keras.backend.concatenate([x_unlabeled, x_labeled], axis=0)
    a_mean, a_log_var, a, z_mean, z_log_var, z = model.encode(x)
    reconstruction = model.decode(a, z)
    
    # Evaluate -log p(x|z,a) for the full mini-batch
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(x,reconstruction), axis=(1, 2)))
    
    # Evaluate KL[q(z|x,a)||p(z)] for the full mini-batch
    kl_loss_z = -0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss_z = tf.reduce_mean(tf.reduce_sum(kl_loss_z, axis=1))
    
    # Evaluate KL[q(a|x)||p(a)] for the unlabeled part of the mini-batch
    kl_loss_a = -0.5*(1 + a_log_var[:unlabeled_batch_size] - \
                                 tf.square(a_mean[:unlabeled_batch_size]) - tf.exp(a_log_var[:unlabeled_batch_size]))
    kl_loss_a = tf.reduce_mean(tf.reduce_sum(kl_loss_a, axis=1))
    
    # free-bits 
    kl_min = 0.5

    kl_loss = kl_weight*(keras.backend.maximum(kl_loss_z, kl_min) + \
        keras.backend.maximum(kl_loss_a, kl_min))
    
    # Evaluate -log q(y|a) for the labeled part of the mini-batch
    a_classifier_inp = a[-labeled_batch_size:]
    y_pred = model.classify(a_classifier_inp)
    # y = tf.cast(y, dtype="float64")
    # y_pred = tf.cast(y_pred, dtype="float64")
    classification_loss_un = keras.losses.SparseCategoricalCrossentropy()(y, y_pred)
    classification_loss = alpha*((unlabeled_batch_size+labeled_batch_size)/labeled_batch_size)*classification_loss_un
    # classification_loss = tf.cast(classification_loss, dtype="float32")

    # L = L_R + kl_weight*(max(L_KLz, kl_min) + max(L_KLy, kl_min)) + alpha*L_CL
    total_loss = reconstruction_loss + kl_loss + classification_loss
    
    return total_loss, reconstruction_loss, kl_loss_z, kl_loss_a, classification_loss_un, kl_loss


@tf.function
def train_step(model, x_labeled, y, x_unlabeled, optimizer, labeled_batch_size, unlabeled_batch_size, kl_weight=1):
    with tf.GradientTape() as tape:
        loss, reconstruction_loss, kl_loss_z, kl_loss_a, classification_loss, kl_loss = compute_loss\
        (model, x_labeled, y, x_unlabeled, labeled_batch_size, unlabeled_batch_size, kl_weight = kl_weight)
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


def clip_values(image_batch):
    iterator = image_batch.flatten()
    for i, element in enumerate(iterator):
        if element <= 1:
            continue
        else:
            iterator[i] = 1
    return np.reshape(iterator, image_batch.shape)  




def label_vae_train(model, unlabeled_dataset, labeled_dataset, y, epochs, labeled_batch_size,\
     unlabeled_batch_size, cost_annealing, channels):
    unlabeled_indices = np.arange(unlabeled_dataset.shape[0])
    # unlabeled_indices = np.delete(unlabeled_indices, labeled_indices)
    labeled_indices = np.arange(labeled_dataset.shape[0])

    for epoch in range(1, epochs+1):
        reset_metrics()

        # cost annealing 
        if cost_annealing:
            if epoch < 10:
                kl_weight = (epoch**3)*0.001
            else:
                kl_weight = 1
        else:
            kl_weight = 1

        
        for training_iteration in range(int(unlabeled_dataset.shape[0]//unlabeled_batch_size)):

            # create unlabeled mini-batch
            x_unlabeled = np.zeros((unlabeled_batch_size, 105, 68, channels))
            for i in range(unlabeled_batch_size):
                index = np.random.choice(unlabeled_indices)
                image = unlabeled_dataset[index]
                x_unlabeled[i] = image

            # create labeled mini-batch
            x_labeled = np.zeros((labeled_batch_size, 105, 68, channels))
            y_mini_batch = np.zeros((labeled_batch_size,))
            for i in range(labeled_batch_size):
                index = np.random.choice(labeled_indices)
                image = labeled_dataset[index]
                x_labeled[i] = image
                y_mini_batch[i] = y[index]

    
            if np.array(x_unlabeled).max() > 1:
                x_unlabeled = clip_values(np.array(x_unlabeled))
            if np.array(x_labeled).max() > 1:
                x_labeled = clip_values(np.array(x_labeled))

            logs = train_step(model, x_labeled, y_mini_batch, x_unlabeled, optimizer,\
                 labeled_batch_size, unlabeled_batch_size, kl_weight = kl_weight)
            
        print('Result at the end of epoch %d:' % (epoch,))
        for key, value in logs.items():
            print('...%s: %.4f' % (key, value))



def train():
    # model architecture
    model = build_architecture()

    # training hyperparameters
    epochs = opt.epochs
    unlabeled_batch_size = opt.unlabeled_batch_size
    labeled_batch_size = opt.labeled_batch_size
    cost_annealing = opt.cost_annealing
    channels = opt.channels

    # data
    labels = opt.labels
    labeled_images = opt.labeled_images
    unlabeled_images = opt.unlabeled_images
    
    # weights
    weight_path = opt.weight_path
    

    # training dataset
    y = np.load(labels, mmap_mode='r')
    labeled_dataset = np.load(labeled_images, mmap_mode='r')#.astype('float32')
    unlabeled_dataset = np.load(unlabeled_images, mmap_mode='r')#.astype('float32)
    
    # model training
    label_vae_train(model, unlabeled_dataset, labeled_dataset, y, epochs, labeled_batch_size,\
         unlabeled_batch_size, cost_annealing, channels)
    model.save_weights(weight_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature-dim', type=int, default=300) 
    parser.add_argument('--hidden-dim', type=int, default=16) # per latent variable
    parser.add_argument('--image-0', type=int, default=105)
    parser.add_argument('--image-1', type=int, default=68)
    parser.add_argument('--channels', type=int, choices=[3,9], default=9)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--unlabeled-batch-size', type=int, default=64)
    parser.add_argument('--labeled-batch-size', type=int, default=4)
    parser.add_argument('--cost-annealing', type=bool, default=True)

    # data + weights
    parser.add_argument('--labels', type=str, required=True, help ='Path to associated labels')
    parser.add_argument('--labeled-frames', type=str, required=True, help='Path to labeled frames')
    parser.add_argument('--unlabeled-frames', type=str, required=True, help='Path to game data')
    parser.add_argument('--weights', type=str, required=True, help='Location to save weights')
    
    opt = parser.parse_args()

    train()
