import numpy as np
from numpy.linalg.linalg import norm

# Generate symmetric matrix
mat = np.random.uniform(low=0, high=1, size=(10, 10))
print(mat.shape)
symm_mat = mat + mat.T
inv_symm_mat = np.linalg.inv(symm_mat)
eig_val_arr, eig_vec_arr = np.linalg.eig(symm_mat)
inv_eig_val_arr, inv_eig_vec_arr = np.linalg.eig(inv_symm_mat)


test_vec = np.random.uniform(low=0, high=1, size=(10, 1))

print('True Norm '+str(np.linalg.norm(symm_mat, ord=2)))
test_vec_orig = test_vec
test_vec_inv = test_vec
for i in range(30):
    test_vec_orig = np.matmul(symm_mat, test_vec_orig)
    test_vec_inv = np.linalg.solve(symm_mat, test_vec_inv)
    test_vec_orig = test_vec_orig / np.linalg.norm(test_vec_orig)
    test_vec_inv = test_vec_inv / np.linalg.norm(test_vec_inv)
    norm_approx_G = np.matmul(np.transpose(test_vec_orig),np.matmul(symm_mat, test_vec_orig)) 
    print(norm_approx_G)

norm_approx_G = np.matmul(np.transpose(test_vec_orig),np.matmul(symm_mat, test_vec_orig)) 
norm_approx_inv_G = np.matmul(np.transpose(test_vec_inv),np.matmul(inv_symm_mat, test_vec_inv)) 
print(norm_approx_G * norm_approx_inv_G)
#print(norm_approx_G)
#print(norm_approx_inv_G)

# eig_val_arr = np.abs(eig_val_arr)
# eig_val_arr = np.sort(eig_val_arr)
# print(eig_val_arr)

inv_eig_val_arr = np.abs(inv_eig_val_arr)
inv_eig_val_arr = np.sort(inv_eig_val_arr)

print(eig_val_arr)
print(np.linalg.norm(symm_mat, ord=2) * np.linalg.norm(inv_symm_mat, ord=2))




