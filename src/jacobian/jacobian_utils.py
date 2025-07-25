import numpy as np
from parsing_utils import to_vector

def perturb(input, delta):
    N = input.shape[0]
    out = np.zeros(shape=(N, N, 2))
    for i in range(N):
        # delta_vec = [0, 0, ..., delta, ..., 0, 0] s.t. delta is at index i 
        delta_vec = np.zeros(shape=(input.shape[0]))
        delta_vec[i] = delta
        # Perturb input by delta
        out[i,:,0] = input - delta_vec
        out[i,:,1] = input + delta_vec
    return out

def partial(out_right, out_left, delta):
    return (out_right - out_left) / (2 * delta)

def jacobian(out_right, out_left, delta):
    #out shapes should be [num_outputs, num_inputs]
    assert out_left.shape == out_right.shape

    M = out_left.shape[0]
    N = out_left.shape[1]

    J = np.zeros(shape=(M,N))
    for i in range(M):
        for j in range(N):
            partial_ij = partial(out_right[i,j], out_left[i,j], delta)
            J[i,j] = partial_ij
   
    return J