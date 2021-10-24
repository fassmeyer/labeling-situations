# Models

The repo contains the presented VAE-based feature extraction methods.
- *SoccerVAE*: a vanilla VAE implementation adapted to the soccer data at-hand
- *LabelVAE*: an enhanced version of the SoccerVAE to foster discriminative causes of variation in the inferred latent feature representations
- *SeqSoccerVAE*: a sequential extension of the SoccerVAE architecture
- *SeqLabelVAE*: a sequential extension of the LabelVAE architecture

Before executing the training scripts, make sure to have preprocessed the datasets as described in the paper and stored as a ```.npy``` file. 



## SoccerVAE
### Training 
```
python train.py --dataset <dataset> --weights <save-weights>
```

Arguments:
- `--intermediate_dim`: Dimension of the hidden layers. Default is '128'.
- `--hidden_dim`: Dimension of the latent variables. Default is '32'.
- `--channels`: Number of channels in image representation. Choices: 9 | 3. Default is '9'.
- `--epochs`: Training epochs. Default is '30'.
- `--batch_size`: Mini-batch size. Default is '32'.
<br>

- `--dataset`: Path to training/game data. 
- `--weights`: File name for the learned parameters of the model. 




## LabelVAE
```
python train.py --unlabeled_frames <dataset> --labeled_frames <labeled-frames> --labels <labels> --weights <save-weights>
```

Main arguments:
- `--feature_dim`: Dimension of the hidden layers. Default is '300'.
- `--hidden_dim`: Dimension of the latent variables. Default is '16'.
- `--channels`: Number of channels in image representation. Choices: 9 | 3. Default is '9'.
- `--epochs`: Training epochs. Default is '30'.
- `--unlabeled_batch_size`: Mini-batch size of the unlabeled data. Default is '64'.
- `--labeled_batch_size`: Mini-batch size of the labeled data. Default is '4'. 
- `--cost_annealing`: Warm-up the KLD in the early training epochs. Default is 'True'.
- `--num_classes`: Number of classes for the auxiliary classifier. Default is '5'.
<br>
- `--unlabeled_frames`: Path to unlabeled training/game data. 
- `--labeled_frames`: Path to labeled training/game data.
- `--labels`: The labels assigned to the labeled_frames data.
- `--weights`: File name for the learned parameters of the model. 




## SeqSoccerVAE
```
python train.py --dataset <dataset> --weights <labeled-frames>
```

Main arguments:
- `--feature_dim`: Dimension of the feature representation for a single frame. Default is '300'.
- `--intermediate_dim`: Dimension of the hidden layers. Default is '512'.
- `--hidden_dim`: Dimension of the latent variables. Default is '16'.
- `--channels`: Number of channels in image representation. Choices: 9 | 3. Default is '9'.
- `--epochs`: Training epochs. Default is '30'.
- `--timesteps`: Sequence length. Default is '20'.
- `--batch_size`: Mini-batch size. Default is '32'. 
<br>
- `--dataset`: Path to training/game data. 
- `--weights`: File name for the learned parameters of the model.  





## SeqLabelVAE
```
python train.py --unlabeled_frames <dataset> --labeled_frames <labeled-frames> --labels <labels> --weights <save-weights>
```


Main arguments:
- `--feature_dim`: Dimension of the feature representation for a single frame. Default is '300'.
- `--intermediate_dim`: Dimension of the hidden layers. Default is '128'.
- `--hidden_dim`: Dimension of the latent variables. Default is '8'.
- `--channels`: Number of channels in image representation. Choices: 9 | 3. Default is '9'.
- `--timesteps`: Sequence length. Default is '20'.
- `--epochs`: Training epochs. Default is '30'.
- `--timesteps`: Sequence length. Default is '20'.
- `--batch_size`: Mini-batch size. Default is '32'. 
- `--unlabeled_batch_size`: Mini-batch size of the unlabeled data. Default is '64'.
- `--labeled_batch_size`: Mini-batch size of the labeled data. Default is '4'. 
- `--num_classes`: Number of classes for the auxiliary classifier. Default is '5'.
- `--cost_annealing`: Warm-up the KLD in the early training epochs. Default is 'False'.
<br>
- `--unlabeled_frames`: Path to unlabeled training/game data. 
- `--labeled_sequences`: Path to labeled sequences. 
- `--labels`: The labels assigned to the labeled_sequences data.
- `--weights`: File name for the learned parameters of the model. 



## Detection 
To obtain low-dimensional vector representations, load the learned parameters and execute forward propagation for the annotated frames/sequences (and the entire test game). Then, for example, an SVM can be trained on the extracted data, which is subsequently used to obtain probability values for the game of interest. We refer to the paper for more details.
