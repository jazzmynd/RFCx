import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from specinput import params

if os.path.exists('./class-freqs.npy'):
    spf = np.load('./class-freqs.npy')
    spf = {i:[j,k] for (i,j,k) in spf}
else:
    print('Cannot find class-freqs.npy. Please generate the file using get-class-frequencies.ipynb')


def get_files_and_labels(data_dir, train_split=None, classes=None, random_state=None):
    """Gathers file paths for each class in a directory and optionally splits them into train/test portions
    
    Expected format:
    
        data_dir/
            class_1/
                file_1
                file_2
                ...
                file_n
            class_2/
            ...
            class_m
    
    Args:
        data_dir: path to data folder
        train_split: float within [0.0, 1.0] determing the portion of each class' samples reserved for training
        classes (optional): list of a subset of classes to collect files for
        random_state (optional): integer rng seed for reproducibility
    
    Returns:
        files_train: list of file paths for training
        files_val: list of file paths for testing
        labels: dictionary mapping classes to label indices for input to a DataGenerator
        
    """
    if classes is None:
        classes = sorted(os.listdir(data_dir))
    files_train = list()
    labels = dict()
    files_val = list()
    for cnt, i in enumerate(classes): # loop over classes
        labels[i]=cnt
        if not os.path.exists(data_dir+"/"+i):
            continue
        files = [data_dir+i+'/'+j for j in os.listdir(data_dir+i)]
        if train_split<1.0:
            tmp_train, tmp_val = train_test_split(files,
                                                  train_size=train_split,
                                                  random_state=random_state)
        else:
            tmp_train = files
            tmp_val = []
        files_train+=tmp_train
        files_val+=tmp_val
                
    return files_train, files_val, labels


class DataGenerator(Sequence):
    """Generates spectrogram batches
    
    Args:
        list_IDs: list of paths to spectrogram files
        labels: dictionary mapping classes to label indices
        dim: (tuple) dimensions of input spectrograms
        resize_dim: (tuple) desired shape of generated images
        batch_size: (int) number of samples per batch
        augment: (boolean) whether to apply data augmentation
        shuffle: (boolean) whether to shuffle the training data after each epoch
        
    """
    def __init__(self, 
                 list_IDs, 
                 labels, 
                 dim = (224, 393), # shape of a single patch
                 resize_dim = None,
                 batch_size=1, 
                 augment=False,
                 shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.dim = dim
        self.resize_dim = resize_dim
        self.list_IDs = list_IDs
        self.n_classes = len(labels)
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data
        
        Args:
            index: index of the desired batch
            
        Returns:
            X: images [<# samples>, <# rows>, <# columns>, <# channels>]
            y: labels [<# samples>, <# classes>]
        
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data  
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        # Initialize image and label arrays  
        if self.resize_dim:
            X = np.empty((self.batch_size, *self.resize_dim, 3)) 
        else:
            X = np.empty((self.batch_size, *self.dim, 3))
        y = np.empty((self.batch_size, self.n_classes))
        class_ids = []

#         y[:] = np.nan # makes no assumption - must use masked loss
        y[:] = 0 # assumes absence of unlabeled classes - use binary cross entropy
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
                                           
            class_id = ID.split('/')[-2]
            class_ids.append(class_id)
            if 'p' in ID.split('/')[-3]:
                y[i,self.labels[class_id]] = 1
#             else: # comment out to assume absence of unlabeled classes
#                 y[i,self.labels[class_id]] = 0
            # print(ID)        
            sample = np.load(ID, allow_pickle=True)
            
            # Random time shift up to 1 second
            # Random frequency shift up to 1%
            sample_frames =int(params.sample_seconds/params.stft_hop_seconds)
            if self.augment:
                shft = int(np.random.choice(range(0, int(2/params.stft_hop_seconds)),1)[0])
                sample = sample[:,shft:(shft+sample_frames),:]
                sample = np.roll(sample, np.random.choice(int(sample.shape[0]*0.01)), axis=0) 
            else:
                sample = sample[:,int(1/params.stft_hop_seconds):(int(1/params.stft_hop_seconds)+sample_frames),:]
            
            # dB-scaling
            sample = scalespec(sample)        
        
            # Image preprocessing
            sample = preprocess(tf.convert_to_tensor(sample), resize_dim=self.resize_dim)  
                        
            X[i,] = sample
            
        if self.augment:
            if np.random.uniform(0,1)<0.5:
                X, y = mixup(X, y, prob=1)
            else:
                X, y = cutmix(X, y, class_ids, prob=1)
                
        return X, y


def scalespec(x, log_offset=0.001):
    """Applies log-scaling to an input amplitude spectrogram
    
    Args:
        log_offset: offset to add to the input spectrogram to avoid log(0)
    
    Returns:
        x: log-scaled spectrogram
        
    """
    return 10*np.log10(x+log_offset)


def preprocess(image, resize_dim=(224, 224)):
    """Preprocesses a greyscale image (spectrogram) for input to a pre-trained ImageNet CNN
    
    Args:
        resize_dim: desired shape of output image
    
    Returns:
        image: pre-processed image
        
    """
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, resize_dim)
    image = tf.image.per_image_standardization(image)
    return image


def mixup(X, y, prob = 1.0):
    """Applies mixup augmentation to a batch of images
    
    Mixup augmenation from: https://arxiv.org/pdf/1710.09412.pdf
    
    Args:
        prob: probability of applying mixup per sample
    
    Returns:
        X: batch with mixup applied
        y: labels with mixup applied
        
    """
    # input X - is a batch of images of size [n,height,width,channels]
    # output - a batch of images with cutmix applied
    batch = np.copy(X)
    labs = np.copy(y)
    for j in range(batch.shape[0]):
        
        # do mixup with specified probability
        if tf.random.uniform([],0,1)<prob:
        
            # pick another sample
            idx = j
            while idx==j:
                idx = tf.random.uniform([],0,batch.shape[0],dtype=tf.int32)
            
            # combine
            a = tf.random.uniform([],0.0,1.0)
            batch[j,] = X[j,]*a + X[idx,]*(1-a)
            
            labs[j,] = y[j,]*a + y[idx,]*(1-a)
            
    return batch, np.stack(labs)


def cutmix(X, y, ids, prob = 1.0):
    """Applies cutmix augmentation to a batch of images
    
    Cutmix augmentation inspired by: https://arxiv.org/abs/1905.04899
    
    This augmentation cuts out the frequency band of a labeled species, and mixes only that band with another sample. This implementation also mixes using the maximum value at each pixel.
    
    Args:
        prob: probability of applying mixup per sample
    
    Returns:
        X: batch with mixup applied
        y: labels with mixup applied
    """
    # input X - is a batch of batchs of size [n,height,width,channels]
    # output - a batch of batchs with cutmix applied
    batch = np.copy(X)
    labs = np.copy(y)
    for j in range(batch.shape[0]):
        
        # do cutmix with specified probability
        if tf.random.uniform([],0,1)<prob:
        
            # pick another sample
            idx = j
            while idx==j:
                idx = tf.random.uniform([],0,batch.shape[0],dtype=tf.int32)
            
            # combine
            batch[j, int(spf[ids[idx]][0]):int(spf[ids[idx]][1])+1, :] = np.maximum(X[j, int(spf[ids[idx]][0]):int(spf[ids[idx]][1])+1, :], X[idx, int(spf[ids[idx]][0]):int(spf[ids[idx]][1])+1, :])
            
            labs[j,] = np.maximum(y[j,], y[idx,])
            
    return batch, np.stack(labs)

    

    

