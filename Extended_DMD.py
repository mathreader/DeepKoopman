import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float64')
import time
import datetime
from scipy import io

data_name = 'Pendulum5'
experiment_tag = 'experiment_31'
input_size = 2 #size of input vector to network
len_time = 2
num_shifts = len_time - 1
num_observables = 400;
width = 800;
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

        self.linear_1 = Linear(input_size, width, title='1')
        self.linear_2 = Linear(width, width, title='2')
        self.linear_3 = Linear(width, width, title='3')
        self.linear_4 = Linear(width, num_observables, title='4')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        x = self.linear_3(x)
        x = tf.nn.relu(x)
        
        return self.linear_4(x)

# The loss function to be optimized
def loss(model, inputs, K):
    # define regularization constants
    lambda_cond = 0.01
    lambda_SL = 10.
    lambda_res = 1.

    # define input data
    layer1 = tf.transpose(inputs[0, :, :])
    layer2 = tf.transpose(inputs[1, :, :])

    # compute G
    #X_data = np.expand_dims(inputs[0, :, :], axis=-1)
    Theta_X = tf.squeeze(model(layer1))
    Theta_Y = tf.squeeze(model(layer2))

    G = tf.linalg.matmul(Theta_X, tf.transpose(Theta_X))
    A = tf.linalg.matmul(Theta_X, tf.transpose(Theta_Y))
    L = tf.linalg.matmul(Theta_Y, tf.transpose(Theta_Y))
    H = tf.linalg.matmul(Theta_Y, tf.transpose(Theta_X))
    
    cond_num = approx_cond_num(G, 30)
    spectral_leakage = spectral_leakage_loss(G, A, L, 30)
    res_loss = lambda_res * residual_loss(G, A, L, H)

    # define loss
    #prediction_error = tf.reduce_mean(tf.square(tf.norm(model(layer2) - tf.linalg.matmul(K,model(layer1)), ord='euclidean', axis=1)))
    prediction_error = tf.reduce_mean(tf.norm(model(layer2) - tf.linalg.matmul(K,model(layer1)), ord=2, axis=1))
    #error3 = lambda_cond*tf.norm(G_new, axis=[-2, -1], ord=2)*tf.norm(tf.linalg.inv(G_new), axis=[-2, -1], ord=2)
    cond_num_error = lambda_cond * (cond_num - 1)
    SL_error = lambda_SL * spectral_leakage

    return prediction_error, cond_num_error, SL_error, res_loss

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

def residual_loss(G, A, L, H):
    # find Extended DMD approximation of K
    K_EDMD = tf.linalg.matmul(tf.linalg.pinv(G),A);

    # find eigenvalues and eigenvectors of K_EDMD
    lambdas, gs = tf.linalg.eig(K_EDMD);

    #cast tensors to data type complex128 (otherwise we can't multiply by complex eigenvalues)
    G = tf.cast(G, tf.complex128)
    A = tf.cast(A, tf.complex128)
    L = tf.cast(L, tf.complex128)
    H = tf.cast(H, tf.complex128)

    res_loss = 0;

    #find residual (squared) for each eigenvalue, eigenvector pair
    for i in range(0, num_observables):
        #cast eigenvector to 10x1 tensor for matrix multiplication purposes
        g = tf.reshape(gs[:,i], (num_observables,1))


        num_matrix = L - lambdas[i]*H - tf.math.conj(lambdas[i])*A + tf.cast(tf.math.abs(lambdas[i]), tf.complex128)*G
        numerator = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(tf.math.conj(g)), num_matrix), g)
        denomenator = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(tf.math.conj(g)), G), g)
        res_loss = res_loss + numerator/denomenator

    return tf.squeeze(tf.math.abs(res_loss))

def grad(model, inputs, K):
    with tf.GradientTape() as tape:
        prediction_error, cond_num_error, SL_error = loss(model, inputs, K)
        loss_value = prediction_error + cond_num_error + SL_error

    return tape.gradient(loss_value, [model.linear_1.w, model.linear_1.b, model.linear_2.w, 
        model.linear_2.b, model.linear_3.w, model.linear_3.b, model.linear_4.w, model.linear_4.b, K])

# Define network model
model = MLPBlock()
model.built = True
#load weights
model.load_weights('./DeepDMD_Weights/weights_{}'.format(experiment_tag))
K_deep = np.load('./DeepDMD_Weights/K_{}.npy'.format(experiment_tag)) #load K from deep DMD loss

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
G_new = np.matmul(tf.transpose(Theta_X),Theta_X)
A_new = np.matmul(tf.transpose(Theta_X),Theta_Y)
L_new = np.matmul(tf.transpose(Theta_Y),Theta_Y)


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

# print K, G, A, L, and the eigenvalues of K to a matlab data file
matlab_dict = {'eigenvalues': lambda_eDMD, 'K':K_dmd, 'G':G_new, 'A':A_new, 'L':L_new}
matlab_file_name = './MatlabFiles/{}_matrices.mat'.format(experiment_tag)
io.savemat(matlab_file_name, matlab_dict)

print("type of lambdas: ")
print(K_deep)
print(K_dmd)

print('Complete eig comp for K_dmd')

#print eigenvalues from largest to smallest
lambda_deepDMD_sorted = np.sort(lambda_deepDMD)
lambda_eDMD_sorted = np.sort(lambda_eDMD)
lambda_deepDMD_sorted = np.absolute(lambda_deepDMD_sorted)
lambda_eDMD_sorted = np.absolute(lambda_eDMD_sorted)
print(lambda_eDMD_sorted.dtype)
print('Complete eig comp for lambda_DMD')

#print eigenvalues
print("Eigenvalues:")
for i in range(num_observables):
	print("Deep DMD: {}, Extended DMD: {}".format(lambda_deepDMD[i], lambda_eDMD[i]))

print("Norm of Deep DMD K = {}".format(tf.linalg.norm(K_deep, axis=[-2, -1], ord=2)))
print("Norm of Extended DMD K = {}".format(tf.linalg.norm(K_dmd,axis=[-2, -1], ord=2)))
print("Condition number of Extended DMD G = {}".format(np.linalg.cond(G_new)))
print("Frobenius Norm Difference of K = {}".format(np.linalg.norm(K_deep - K_dmd)))
