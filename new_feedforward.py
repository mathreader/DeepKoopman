import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float64')

data_name = 'Pendulum'
len_time = 51
num_shifts = len_time - 1

# Function for stacking the data
def stack_data(data, num_shifts, len_time):
    """Stack data from a 2D array into a 3D array.

    Arguments:
        data -- 2D data array to be reshaped
        num_shifts -- number of shifts (time steps) that losses will use (maximum is len_time - 1)
        len_time -- number of time steps in each trajectory in data

    Returns:
        data_tensor -- data reshaped into 3D array, shape: num_shifts + 1, num_traj * (len_time - num_shifts), n

    Side effects:
        None
    """
    nd = data.ndim
    if nd > 1:
        n = data.shape[1]
    else:
        data = (np.asmatrix(data)).getT()
        n = 1
    num_traj = int(data.shape[0] / len_time)

    new_len_time = len_time - num_shifts

    data_tensor = np.zeros([num_shifts + 1, num_traj * new_len_time, n])

    for j in np.arange(num_shifts + 1):
        for count in np.arange(num_traj):
            data_tensor_range = np.arange(count * new_len_time, new_len_time + count * new_len_time)
            data_tensor[j, data_tensor_range, :] = data[count * len_time + j: count * len_time + j + new_len_time, :]

    return data_tensor

# Process Data
data_orig = np.loadtxt(('./data/%s_train1_x.csv' % (data_name)), delimiter=',', dtype=np.float64)
data_val = np.loadtxt(('./data/%s_val_x.csv' % (data_name)), delimiter=',', dtype=np.float64)

data_orig_stacked = stack_data(data_orig, num_shifts, len_time)
data_val_stacked = stack_data(data_val, num_shifts, len_time)

# Custom Linear Layer
class Linear(keras.layers.Layer):
    def __init__(self, input_dim=32, output_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(output_dim, input_dim), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(output_dim,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.math.add(tf.matmul(self.w, inputs), tf.expand_dims(self.b, 1))

# Create model
class MLPBlock(keras.layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(2, 80)
        self.linear_2 = Linear(80, 80)
        self.linear_3 = Linear(80, 80)
        self.linear_4 = Linear(80, 80)
        self.linear_5 = Linear(80, 2)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        x = self.linear_3(x)
        x = tf.nn.relu(x)
        x = self.linear_4(x)
        x = tf.nn.relu(x)
        return self.linear_5(x)

# The loss function to be optimized
def loss(model, inputs, num_loss_steps):
    initial_layer = tf.transpose(inputs[0, :, :])
    current_layer = initial_layer
    error = 0
    for i in range(num_loss_steps):
        # Compute the network output after i iterations
        current_layer = model(current_layer)
        if (i == 0):
            error = tf.reduce_mean(tf.reduce_mean(tf.square(current_layer - tf.transpose(inputs[i+1, :, :])), axis=0))
        else: 
            error = error + tf.reduce_mean(tf.reduce_mean(tf.square(current_layer - tf.transpose(inputs[i+1, :, :])), axis=0))
    
    error = error / num_loss_steps
    return error

def grad(model, inputs, num_loss_steps):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, num_loss_steps)
    return tape.gradient(loss_value, [model.linear_1.w, model.linear_1.b, model.linear_2.w, 
        model.linear_2.b, model.linear_3.w, model.linear_3.b, model.linear_4.w, model.linear_4.b, model.linear_5.w, model.linear_5.b])


# Define network model
model = MLPBlock()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
print("weights:", len(model.weights))
print("trainable weights:", len(model.trainable_weights))

# Weights of the model is given by model.linear1.w, model.linear1.b, model.linear2.w, model.linear2.b

# The loss function to be optimized
steps = 100
for i in range(steps):
    # Training step
    grads = grad(model, data_orig_stacked, 50)
    optimizer.apply_gradients(zip(grads, [model.linear_1.w, model.linear_1.b, model.linear_2.w, 
        model.linear_2.b, model.linear_3.w, model.linear_3.b, model.linear_4.w, model.linear_4.b, model.linear_5.w, model.linear_5.b]))
    
    if i % 20 == 0:
        # Evaluation step
        print("Step-1 Training Loss at step {:03d}: {:.3f}".format(i, loss(model, data_orig_stacked, 1)))
        print("Step-1 Evaluation Loss at step {:03d}: {:.3f}".format(i, loss(model, data_val_stacked, 1)))
        print("Step-50 Training Loss at step {:03d}: {:.3f}".format(i, loss(model, data_orig_stacked, 50)))
        print("Step-50 Evaluation Loss at step {:03d}: {:.3f}".format(i, loss(model, data_val_stacked, 50)))
