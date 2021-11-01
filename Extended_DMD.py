import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float64')
import time
import datetime
import cvxpy as cp

data_name = 'Pendulum'
len_time = 51
num_shifts = len_time - 1
num_observables = 10;
lambda1 = 0.01

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

print(data_orig_stacked.shape)

# Custom Linear Layer
class Linear(keras.layers.Layer):
    def __init__(self, input_dim=32, output_dim=32, title=''):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(output_dim, input_dim), initializer="random_normal", trainable=True, regularizer='l2', name = 'W'+title)
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
        
        return self.linear_3(x)

# The loss function to be optimized
def loss(model, inputs, K):
    # define regularization constants
    lambda1 = 0.01
    lambda_G = (10**-3)/num_observables # divide by num_observables since the Frobenius norm scales with the size of the matrix

    # define input data
    layer1 = tf.transpose(inputs[0, :, :])
    layer2 = tf.transpose(inputs[1, :, :])

    # compute G
    #X_data = np.expand_dims(inputs[0, :, :], axis=-1)
    Theta_X = np.squeeze(model(layer1))
    G_new = np.matmul(Theta_X,np.transpose(Theta_X))

    # define loss
    error = tf.reduce_mean(tf.norm(model(layer2) - tf.linalg.matmul(K,model(layer1)), ord=2, axis=1)) + lambda_G*tf.norm(G_new - np.identity(num_observables))
    return error

def grad(model, inputs, K):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, K)
    return tape.gradient(loss_value, [model.linear_1.w, model.linear_1.b, model.linear_2.w, 
        model.linear_2.b, model.linear_3.w, model.linear_3.b, K])


# Define network model
model = MLPBlock()
model.built = True
#load weights
model.load_weights('./DeepDMD_Weights/weights_experiment_6_10')
K_deep = np.load('./DeepDMD_Weights/K_experiment_6_10.npy') #load K from deep DMD loss

## compute K_dmd using extended DMD with the neural network as the dictionary functions
#construct matrix Theta
# num_seq = tf.shape(data_orig_stacked).numpy()[1]
# Theta_X = np.zeros((num_observables, num_seq))
# Theta_Y = np.zeros((num_observables, num_seq))

# for i in range(num_seq):
#     Theta_X[:,i] = np.squeeze(model.predict(data_orig_stacked[0,i,:]))
#     Theta_Y[:,i] = np.squeeze(model.predict(data_orig_stacked[1,i,:]))
#     print("loop number {}".format(i))

X_data = np.expand_dims(data_orig_stacked[0,:,:], axis=-1)
Y_data = np.expand_dims(data_orig_stacked[1,:,:], axis=-1)
print(X_data.shape)
print(Y_data.shape)
Theta_X = np.squeeze(model.predict(X_data))
Theta_Y = np.squeeze(model.predict(Y_data))
print(Theta_X.shape)
print(Theta_Y.shape)


# Regularized Version suggested in "learning deep neural network representations for koopman operators of nonlinear dynamical systems" 
# Solve optimization problem via CVXPY
#K_solve = cp.Variable((num_observables, num_observables), complex=True)
#cost = cp.norm2(cp.transpose(Theta_Y[1:100, :]) - K_solve @ cp.transpose(Theta_X[1:100, :])) + lambda1 * cp.norm2(K_solve)
#prob = cp.Problem(cp.Minimize(cost))
#prob.solve(verbose=True)

#K_dmd = K_solve.value


# Implementation in the Paper "A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition"
print('Computing inverse')
G_new = np.matmul(np.transpose(Theta_X),Theta_X)
A_new = np.matmul(np.transpose(Theta_X),Theta_Y)
print(G_new.shape)
print(A_new.shape)
inv_G = np.linalg.pinv(G_new)
K_dmd = np.matmul(inv_G, A_new)
print('Compute K-dmd complete')
print(K_dmd.shape)

# Implementation in the Book "Data-Driven Science and Engineering Machine Learning, Dynamical Systems, and Control"
# print('Computing inverse')
# inv_X = np.linalg.pinv(np.transpose(Theta_X))
# print(inv_X.shape)
# print('Computing K-dmd')
# K_dmd = np.matmul(np.transpose(Theta_Y), inv_X)
# print('Compute K-dmd complete')
# print(K_dmd.shape)
# print(K_dmd)

## compute eigenvalues and eigenvectors of K and K_dmd
lambda_deepDMD, v_deepDMD = np.linalg.eig(K_deep)
lambda_eDMD, v_eDMD = np.linalg.eig(K_dmd)
print(K_deep)
print(K_dmd)

print('Complete eig comp for K_dmd')

#print eigenvalues from largest to smallest
lambda_deepDMD_sorted = np.sort(lambda_deepDMD)
lambda_eDMD_sorted = np.sort(lambda_eDMD)
lambda_deepDMD_sorted = np.absolute(lambda_deepDMD_sorted)
lambda_eDMD_sorted = np.absolute(lambda_eDMD_sorted)

print('Complete eig comp for lambda_DMD')

#print eigenvalues
print("Eigenvalues:")
for i in range(num_observables):
	print("Deep DMD: {}, Extended DMD: {}".format(lambda_deepDMD_sorted[i], lambda_eDMD_sorted[i]))

print("Norm of Deep DMD K = {}".format(tf.linalg.norm(K_deep,ord=2)))
print("Norm of Extended DMD K = {}".format(tf.linalg.norm(K_dmd,ord=2)))
print("Condition number of Extended DMD G = {}".format(np.linalg.cond(G_new)))
print("Frobenius Norm Difference G = {}".format(np.linalg.norm(K_deep - K_dmd)))
