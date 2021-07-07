import numpy as np
import tensorflow as tf
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# define constants
seed = 5; #set seed to a constant for reproducibility when running experiments
BatchSize = 128;
SHUFFLE_BUFFER_SIZE = 100;
max_time = 20; # time to train for, in minutes
data_name = 'Pendulum'
len_time = 51

# Setup the seed for the experiment
tf.set_random_seed(seed)
np.random.seed(seed)

# Data Processing Functions

# Now we would like to structure the data in the following way:
# Loop: data_orig = [x1, x2, ..., x51] -> data_x = [x1, x2, ..., x50] / data_y = [x2, x3, ..., x51]
def data_generate(data_orig):
    '''Transforms data into our learning task of predicting next step from current step.'''
    
    # Initialize size
    data_size = len(data_orig)
    num_iters = int(np.floor(data_size / len_time))
    
    # Initialize new datasets
    data_x = np.zeros(((len_time-1)*num_iters, 2))
    data_y = np.zeros(((len_time-1)*num_iters, 2))
    
    # Loop to generate new datasets
    for i in range(num_iters):
        input_index_start = i * len_time 
        output_index_start = i * (len_time - 1)
        
        for j in range(len_time - 1):
            data_x[output_index_start + j, :] = data_orig[input_index_start + j, :]
            data_y[output_index_start + j, :] = data_orig[input_index_start + j + 1, :]
    
    return data_x, data_y

def data_batch_generate(data_x, data_y, batch_size):
    '''Reshapes data to be in batches. Each 2d tensor corresponding to data_batch_x[:,i,:]
    or data_batch_y[:,i,:] (for a given i) contains a single batch of data. '''

    # determine number of batches. 
    #We round down, which effectively cuts off some of the data if its length is not an integer multiple of the batch size.
    num_batches = int(np.floor(data_x.shape[0]/batch_size))
    
    data_batch_x = np.zeros((batch_size, num_batches, 2))
    data_batch_y = np.zeros((batch_size, num_batches, 2))
    
    # copy data to data_batch_x and data_batch_y
    for i in range(num_batches):
        data_batch_x[:,i,:] = data_x[i*batch_size:(i+1)*batch_size,:]
        data_batch_y[:,i,:] = data_y[i*batch_size:(i+1)*batch_size,:]
        
    return data_batch_x, data_batch_y


def data_extract_first_entry(data_orig):
    '''Extracts only first entry in the data in each sequence for comparison to DeepKoopman'''
    
    # Initialize size
    data_size = len(data_orig)
    num_iters = int(np.floor(data_size / len_time))
    
    # Initialize dataset with first entry
    data_x = np.zeros((num_iters, 2))
    data_y = np.zeros((num_iters, 2))
    
    # Only put first entry in dataset
    
    for i in range(num_iters):
        input_index_start = i * len_time 
        
        data_x[i, :] = data_orig[i * len_time, :]
        data_y[i, :] = data_orig[i * len_time + 1, :]
    
    return data_x, data_y


# Define loss function
def custom_loss(y_actual,y_pred):
    #custom_loss=tf.keras.backend.mean(tf.math.reduce_sum(tf.square(y_actual-y_pred), axis=-1))
    #length = y_actual.shape[0]
    custom_loss=tf.math.reduce_mean(tf.math.reduce_sum(tf.square(y_actual-y_pred), axis=-1))
    return 

# Define network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(80, activation='relu', input_shape = (2,)),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(2)
])


# Process Data
data_orig = np.loadtxt(('./data/%s_train1_x.csv' % (data_name)), delimiter=',', dtype=np.float64)
data_val = np.loadtxt(('./data/%s_val_x.csv' % (data_name)), delimiter=',', dtype=np.float64)

# Training Data
data_x, data_y = data_generate(data_orig)
train_dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BatchSize)

# Validation Data
data_val_x, data_val_y = data_generate(data_val)
val_dataset = tf.data.Dataset.from_tensor_slices((data_val_x, data_val_y))
val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BatchSize)

# Comparison Data
data_val_comp_x, data_val_comp_y = data_extract_first_entry(data_val)
comp_dataset = tf.data.Dataset.from_tensor_slices((data_val_comp_x, data_val_comp_y))
comp_dataset = comp_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BatchSize)

# initiate network
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=custom_loss)

# train network
start_time = time.time();
epoch_num = 1;
best_error = 10**12;

print("Starting training")
while ((time.time() - start_time) < max_time*60): #multiply max_time by 60 to get time in seconds
    # Train
    model.trainable = True
    model.fit(train_dataset, epochs = 1, verbose = 1)
        
    if epoch_num%10 == 0:
        model.trainable = False
        
        # Compute the error on the training data set
        train_loss = model.evaluate(train_dataset)
        print("Training loss: " + str(train_loss))
        
    
        # Compute the error on the validation set
        val_loss = model.evaluate(val_dataset)
        print("Validation loss: " + str(val_loss))

        # Compute the comparison loss
        val_loss_2 = model.evaluate(comp_dataset)
        print("Comparison loss: " + str(val_loss_2))
        
        # Save weights corresponding to lowest validation loss
        if val_loss_2 < (best_error - best_error * (10 ** (-5))):
            best_error = val_loss_2.copy()
            model.save_weights('./checkpoints/my_checkpoint')
            
    epoch_num = epoch_num + 1;
        
        
print("20 minutes of training complete")