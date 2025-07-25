from parsing_utils import *
from jacobian_utils import *
import numpy as np
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from inference import run_inference, load_config_and_checkpoint
import torch

AVG_KY_LOCS = torch.tensor([0.06010753, 0.12021505, 0.18032258, 0.2404301, 0.30053763, 0.54096774,
  0.66118279, 0.78139784, 0.90161289, 1.02182795, 1.142043, 1.26225805,
  1.20215052, 1.5988144, 2.12677655, 2.82962789, 3.76547314, 5.01177679,
  6.67183152, 8.88339377, 11.8302146, 15.75743919, 20.99217655, 27.97098031]).unsqueeze(0)

def sample_and_perturb(N):
    perturbations = np.zeros(shape=(N, 31, 31, 2))
    samples = np.zeros(shape=(N, 31))
    for i in range(N):
        sample = sample_input('./merged_dist.json', as_vector=False)
        sample_vec = to_vector(sample)
        samples[i,:] = sample_vec
        delta = 0.01
        perturbed = perturb(sample_vec, delta)
        perturbations[i, :, :, :] = perturbed
        if i % 1000 == 0:
            print(f'Generated sample {i}')

    with open("./samples.pkl", "wb") as file:
        # Dump the object to the file
        pickle.dump(samples, file)

    with open("./perturbations.pkl", "wb") as file:
        # Dump the object to the file
        pickle.dump(perturbations, file)

    samples_all = np.zeros(shape = (N * 31 * 2, 31))
    for i in range(N):
        sample = perturbations[i]
        for j in range(31):
            for k in range(2):
                samples_all[(i * 31 * 2) + (j * 2) + k, :] = sample[j, :, k].squeeze()
    return samples_all

def infer_perturbations(inputs):
    input_tensor = torch.tensor(inputs)
    ky_locs = AVG_KY_LOCS.repeat([input_tensor.shape[0], 1])
    ky_locs_expanded = ky_locs[:, :, torch.newaxis]

    input_tensor = input_tensor[:, torch.newaxis, :].repeat([1, 24, 1])
    input_tensor = torch.cat([input_tensor, ky_locs_expanded], dim=-1)

    cfg, ckpt = load_config_and_checkpoint('./')
    pred = run_inference(cfg, ckpt, input_tensor)

    with open("./predictions.pkl", "wb") as file:
        # Dump the object to the file
        pickle.dump(pred, file)

    return pred

def expand_predictions(pred):
    predictions_expanded = np.zeros(shape=(1000, 31, 4, 2))
    for j in range((int)(pred.shape[0] / 62)):
        i = j * 62
        left = pred[i:i+31, :]
        right = pred[i+31:i+62, :]
        predictions_expanded[j, :, :, 0] = left
        predictions_expanded[j, :, :, 1] = right
    return predictions_expanded

def trace(J):
    J_T = J.T
    sq = J_T @ J
    return np.trace(sq)

def compute_jacobians(samples):
    pred = infer_perturbations(samples)
    predictions_expanded = expand_predictions(pred)
    jacobians = np.zeros(shape=(predictions_expanded.shape[0], 31, 4))
    for i in range(predictions_expanded.shape[0]):
        # Compute jacobian for input i
        out_left = predictions_expanded[i,:,:,0]
        out_right = predictions_expanded[i,:,:,1]
        J = jacobian(out_right, out_left, 0.01)
        jacobians[i,:,:] = J

    with open("./jacobians.pkl", "wb") as file:
        # Dump the object to the file
        pickle.dump(jacobians, file)

    traces = np.zeros(shape=(jacobians.shape[0]))
    for i in range(jacobians.shape[0]):
        traces[i] = trace(jacobians[i,:,:])

    with open("./traces.pkl", "wb") as file:
        # Dump the object to the file
        pickle.dump(traces, file)
    
    return traces



if __name__ == "__main__":
    N = 100000
    K = 10000
    samples = sample_and_perturb(N)
    traces = compute_jacobians(samples)
    sorted_idxs = np.argsort(-traces) #argsort is always ascending, so taking -traces is a simple way to get the descending sorted idxs
    topK = sorted_idxs[:K]
    samples_topK = samples[topK]

    with open("./samples_topk.pkl", "wb") as file:
        # Dump the object to the file
        pickle.dump(samples_topK, file)

