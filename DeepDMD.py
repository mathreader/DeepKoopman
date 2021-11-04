import numpy as np
from numpy.linalg.linalg import norm
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float64')
import time
import datetime

data_name = 'Lorenz'
input_size = 3 #size of input vector to network
len_time = 51
num_shifts = len_time - 1
data_file_path = './DeepDMD_results/{}_experiment_10_10_{}_error.csv'.format(data_name, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"))
max_time = 60; #time to run training, in minutes
num_observables = 10;
reg_param = 1e-5


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

        self.linear_1 = Linear(input_size, 80, title='1')
        self.linear_2 = Linear(80, 80, title='2')
        self.linear_3 = Linear(80, 80, title='3')
        self.linear_4 = Linear(80, num_observables, title='4')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        x = self.linear_3(x)
        x = tf.nn.relu(x)
        x = self.linear_4(x)

        # x_scaled = tf.math.divide(x,(1 + inputs[1,:]**2)) #scale x so that output is in L2
        x_scaled = tf.math.divide(x,tf.math.exp(inputs[1,:]**2)) #scale x so that output is in L2
        
        return x_scaled

# The loss function to be optimized
def loss(model, inputs, K):
    # define regularization constants
    lambda_cond = 0.001
    lambda_SL = 10.

    # define input data
    layer1 = tf.transpose(inputs[0, :, :])
    layer2 = tf.transpose(inputs[1, :, :])

    # compute G
    #X_data = np.expand_dims(inputs[0, :, :], axis=-1)
    Theta_X = tf.squeeze(model(layer1))
    Theta_Y = tf.squeeze(model(layer2))
    #print('Matrix Theta_X:')
    #print(Theta_X)

    G = tf.linalg.matmul(Theta_X, tf.transpose(Theta_X))
    A = tf.linalg.matmul(Theta_X, tf.transpose(Theta_Y))
    L = tf.linalg.matmul(Theta_Y, tf.transpose(Theta_Y))
    #print('Matrix G')
    #print(G)
    
    cond_num = approx_cond_num(G, 30)
    spectral_leakage = spectral_leakage_loss(G, A, L, 30)
    #print("Spectral Leakage = {}".format(spectral_leakage))

    # define loss
    #prediction_error = tf.reduce_mean(tf.square(tf.norm(model(layer2) - tf.linalg.matmul(K,model(layer1)), ord='euclidean', axis=1)))
    prediction_error = tf.reduce_mean(tf.norm(model(layer2) - tf.linalg.matmul(K,model(layer1)), ord=2, axis=1))
    #error3 = lambda_cond*tf.norm(G_new, axis=[-2, -1], ord=2)*tf.norm(tf.linalg.inv(G_new), axis=[-2, -1], ord=2)
    cond_num_error = lambda_cond * cond_num
    SL_error = lambda_SL * spectral_leakage

    #print('Error of norm '+str(np.abs(norm_approx_G -tf.norm(G_new))))
    #print('Error of norm inverse '+str(np.abs(norm_approx_inv_G -norm(tf.linalg.inv(G_new)))))
    return prediction_error, cond_num_error, SL_error


def approx_cond_num(G, num_iter):
    # Power iteration
    test_vector = tf.random.uniform((num_observables, 1), minval=0, maxval=1, dtype=tf.dtypes.float64)
    test_vector = test_vector / tf.norm(test_vector)
    
    test_vector_orig = test_vector
    test_vector_inv = test_vector
    for i in range(num_iter):

        test_vector_orig = tf.linalg.matmul(G, test_vector_orig)
        test_vector_inv = tf.linalg.solve(G, test_vector_inv)
        
        test_vector_orig = test_vector_orig / tf.norm(test_vector_orig)
        test_vector_inv = test_vector_inv / tf.norm(test_vector_inv)
        
        #print("Norm True Cond: {:.5e}, Approx Cond {:.5e}, ".format(lambda_cond*tf.norm(G_new, axis=[-2, -1], ord=2)*tf.norm(tf.linalg.inv(G), axis=[-2, -1], ord=2), lambda_cond * norm_approx_G * norm_approx_inv_G))

    
    norm_approx_G = tf.squeeze(tf.linalg.matmul(tf.transpose(test_vector_orig),tf.linalg.matmul(G, test_vector_orig)))
    norm_approx_inv_G = tf.squeeze(tf.linalg.matmul(tf.transpose(test_vector_inv),tf.linalg.matmul(tf.linalg.inv(G), test_vector_inv)))

    return norm_approx_G * norm_approx_inv_G

def spectral_leakage_loss(G, A, L, num_iter):
    test_vector = tf.random.uniform((num_observables, 1), minval=0, maxval=1, dtype=tf.dtypes.float64)
    test_vector = test_vector / tf.sqrt(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(test_vector),G),test_vector))

    for i in range(num_iter):
        w1 = tf.matmul(L, test_vector)
        w2 = tf.linalg.matmul(tf.transpose(A) ,tf.linalg.solve(G,tf.linalg.matmul(tf.transpose(A), test_vector)))
        w = tf.linalg.solve(G, w1 - w2)

        #normalize test vector
        test_vector = w / tf.sqrt(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(w), G), w))

    w1 = tf.matmul(L, test_vector)
    w2 = tf.linalg.matmul(tf.transpose(A) ,tf.linalg.solve(G,tf.linalg.matmul(tf.transpose(A), test_vector)))

    spectral_leakage = (tf.linalg.matmul(tf.transpose(test_vector), w1-w2))/(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(test_vector), G), test_vector))

    return tf.squeeze(spectral_leakage)

def grad(model, inputs, K):
    with tf.GradientTape() as tape:
        prediction_error, cond_num_error, SL_error = loss(model, inputs, K)
        loss_value = prediction_error + cond_num_error + SL_error

    return tape.gradient(loss_value, [model.linear_1.w, model.linear_1.b, model.linear_2.w, 
        model.linear_2.b, model.linear_3.w, model.linear_3.b, K])


# Define network model
model = MLPBlock()
normal_vector = tf.random.normal(
    (num_observables, num_observables), mean=0.0, stddev=1.0, dtype=tf.dtypes.float64, seed=5)
K = tf.Variable(initial_value=normal_vector)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


print("weights:", len(model.weights))
print("trainable weights:", len(model.trainable_weights))
# Weights of the model is given by model.linear1.w, model.linear1.b, model.linear2.w, model.linear2.b

#open file to print data
f = open(data_file_path, 'w')
f.write("Epoch #, Runtime, Train Loss, Val Loss\n")

epoch_num = 1;
start_time = time.time();
best_val_loss = 10**8

while ((time.time() - start_time) < max_time*60):
    # Training step
    grads = grad(model, data_orig_stacked, K)
    optimizer.apply_gradients(zip(grads, [model.linear_1.w, model.linear_1.b, model.linear_2.w, 
        model.linear_2.b, model.linear_3.w, model.linear_3.b, K]))
    
    if (epoch_num-1) % 10 == 0:
        # Evaluation step
        train_prediction_error, train_cond_num_error, train_SL_error    = loss(model, data_orig_stacked, K)
        val_prediction_error, val_cond_num_error, val_SL_error          = loss(model, data_val_stacked, K)

        #print results
        print("Epoch number {}".format(epoch_num))
        print("Training Prediction Loss: {:.5e}".format(train_prediction_error))
        print("Evaluation Prediction Loss: {:.5e}".format(val_prediction_error))

        print("Training G Condition Number: {:.5e}".format(train_cond_num_error))
        print("Evaluation G Condition Number: {:.5e}".format(val_cond_num_error))

        print("Training Spectral Leakage: {:.5e}".format(train_SL_error))
        print("Evaluation Spectral Leakage: {:.5e}".format(val_SL_error))

        if (val_prediction_error < best_val_loss):
            best_val_loss = val_prediction_error
            print("\nNew best prediction loss: {:.5e}\n".format(best_val_loss))

            # save weights and K
            model.save_weights('./DeepDMD_Weights/weights_experiment_10_10')
            np.save('./DeepDMD_Weights/K_experiment_10_10.npy', K.numpy())


        # print loss data to file
        f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(epoch_num, time.time() - start_time, train_prediction_error, train_cond_num_error, train_SL_error, val_prediction_error, val_cond_num_error, val_SL_error))

    epoch_num = epoch_num + 1;

f.close() 

