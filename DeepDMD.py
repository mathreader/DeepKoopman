import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float64')
import time
import datetime

data_name = 'Pendulum'
len_time = 51
num_shifts = len_time - 1
data_file_path = './DeepDMD_results/Pendulum_no_reg{}_error.csv'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
max_time = 60; #time to run training, in minutes
num_observables = 10;

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
    def __init__(self, input_dim=32, output_dim=32, title=''):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(output_dim, input_dim), initializer="random_normal", trainable=True, regularizer='l1', name = 'W'+title)
        self.b = self.add_weight(shape=(output_dim,), initializer="zeros", trainable=True, name = 'b' + title)

    def call(self, inputs):
        return tf.math.add(tf.matmul(self.w, inputs), tf.expand_dims(self.b, 1))

# Create model
class MLPBlock(tf.keras.Model):
    def __init__(self):
        super(MLPBlock, self).__init__()

        self.linear_1 = Linear(2, 40, title='1')
        self.linear_2 = Linear(40, 40, title='2')
        self.linear_3 = Linear(40, num_observables, title='3')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        x = self.linear_3(x)

        # x_scaled = tf.math.divide(x,(1 + inputs[1,:]**2)) #scale x so that output is in L2
        x_scaled = tf.math.divide(x,tf.math.exp(inputs[1,:]**2)) #scale x so that output is in L2
        
        return x_scaled

# The loss function to be optimized
def loss(model, inputs, K):
    # define regularization constants
    lambda1 = 0.01
    lambda_G = (10**-1)/num_observables # divide by num_observables since the Frobenius norm scales with the size of the matrix
    lambda_cond = 0.01

    # define input data
    layer1 = tf.transpose(inputs[0, :, :])
    layer2 = tf.transpose(inputs[1, :, :])

    # compute G
    #X_data = np.expand_dims(inputs[0, :, :], axis=-1)
    Theta_X = tf.squeeze(model(layer1))
    G_new = tf.linalg.matmul(Theta_X,np.transpose(Theta_X))

    # define loss
    error1 = tf.reduce_mean(tf.norm(model(layer2) - tf.linalg.matmul(K,model(layer1)), ord=2, axis=1))
    error2 = lambda_G*tf.norm(G_new - np.identity(num_observables))
    error3 = lambda_cond*tf.norm(G_new)*tf.norm(tf.linalg.inv(G_new))
    return error1, error2, error3


def grad(model, inputs, K):
    with tf.GradientTape() as tape:
        error1, error2, error3 = loss(model, inputs, K)
        loss_value = error1 + error3
    return tape.gradient(loss_value, [model.linear_1.w, model.linear_1.b, model.linear_2.w, 
        model.linear_2.b, model.linear_3.w, model.linear_3.b, K])


# Define network model
model = MLPBlock()
normal_vector = tf.random.normal(
    (num_observables, num_observables), mean=0.0, stddev=1.0, dtype=tf.dtypes.float64, seed=5)
K = tf.Variable(initial_value=normal_vector)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


print("weights:", len(model.weights))
print("trainable weights:", len(model.trainable_weights))
# Weights of the model is given by model.linear1.w, model.linear1.b, model.linear2.w, model.linear2.b

#open file to print data
f = open(data_file_path, 'w')
f.write("Epoch #, Runtime, Train Loss, Val Loss\n")

epoch_num = 1;
start_time = time.time();

while ((time.time() - start_time) < max_time*60):
    # Training step
    grads = grad(model, data_orig_stacked, K)
    optimizer.apply_gradients(zip(grads, [model.linear_1.w, model.linear_1.b, model.linear_2.w, 
        model.linear_2.b, model.linear_3.w, model.linear_3.b, K]))
    
    if (epoch_num-1) % 10 == 0:
        # Evaluation step
        train_error_1, train_error_2, train_error_3    = loss(model, data_orig_stacked, K)
        val_error_1, val_error_2, val_error_3      = loss(model, data_val_stacked, K)

        #print results
        print("Epoch number {}".format(epoch_num))
        print("Training True Loss: {:.5e}".format(train_error_1))
        print("Evaluation True Loss: {:.5e}".format(val_error_1))
        print("Training Regularization Loss: {:.5e}".format(train_error_2))
        print("Evaluation Regularization Loss: {:.5e}".format(val_error_2))
        print("Training G Condition Number: {:.5e}".format(train_error_3))
        print("Evaluation G Condition Number: {:.5e}".format(val_error_3))

        # print loss data to file
        f.write("{}, {}, {}, {}\n".format(epoch_num, time.time() - start_time, train_error_1, train_error_2, train_error_3, val_error_1, val_error_2, val_error_3))

    epoch_num = epoch_num + 1;

f.close() 

#save weights
model.save_weights('./DeepDMD_Weights/weights_no_reg_split_loss_c_0_01')

#save K
np.save('./DeepDMD_Weights/K_no_reg_split_loss_c_0_01.npy', K.numpy())

