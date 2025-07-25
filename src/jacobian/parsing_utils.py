import numpy as np
import os
import json
from pyrokinetics import Pyro

TGLF_CONSTANTS = {
    'SIGN_BT': -1.00000E+00,
    'SIGN_IT': +1.00000E+00,
    'NS':3,
    'ZS_2': 1,
    'MASS_2': +1.00000E+00,
    'ZS_3': 6,
    'MASS_3': +6.00000E+00,
    'ZS_1': -1,
    'MASS_1': +2.72444E-04,
    'VPAR_SHEAR_2':+1.27916E-01,
    'VPAR_2':+1.77710E-01,
    'VPAR_SHEAR_3':+1.27916E-01,
    'VPAR_3':+1.77710E-01,
    'AS_1':+1.00000E+00,
    'TAUS_1':+1.00000E+00,
    'NKY': 24,
    'USE_BPER': True,
    'USE_BPAR': True,
    'USE_AVE_ION_GRID':False,
    'USE_MHD_RULE':False,
    'ALPHA_ZF':-1,
    'KYGRID_MODEL':4,
    'SAT_RULE':3,
    'NBASIS_MAX':6,
    'USE_TRANSPORT_MODEL':True,
    'GEOMETRY_FLAG':1
}

CGYRO_CONSTANTS = {
    'N_ENERGY':8,
    'N_XI':24,
    'N_THETA':24,
    'N_RADIAL':16,
    'N_TOROIDAL':1,
    'NONLINEAR_FLAG':0,
    'BOX_SIZE':1,
    'DELTA_T':0.005,
    'MAX_TIME':100000.0,
    'PRINT_STEP':100,
    'THETA_PLOT': 1
}

KY_LOCS = [0.06010753, 0.12021505, 0.18032258, 0.2404301, 0.30053763, 0.54096774,
  0.66118279, 0.78139784, 0.90161289, 1.02182795, 1.142043, 1.26225805,
  1.20215052, 1.5988144, 2.12677655, 2.82962789, 3.76547314, 5.01177679,
  6.67183152, 8.88339377, 11.8302146, 15.75743919, 20.99217655, 27.97098031]

def get_dists(json_path):
    with open(json_path, 'r') as file:
        input_dists = json.load(file)
    return input_dists

def sample_input(json_path, as_vector=True):
    sampled = {}
    with open(json_path, 'r') as file:
        input_dists = json.load(file)

    for input_name in input_dists:
        dist = input_dists[input_name]
        mean, std = dist['mean'], dist['std']
        sample = np.random.normal(mean, std)
        # if("_log10" in input_name):
        #     input_name = input_name[:-len('_log10')]
        #     sample = 10 ** sample
        sampled[input_name] = sample

    if as_vector:
        return to_vector(sampled)
    return sampled

def to_vector(input_dict):
    vec = np.zeros(shape=(len(input_dict)))
    i = 0
    for key in input_dict:
        vec[i] = input_dict[key]
        i+=1
    return vec

def to_dict(input_vec, like_dict):
    assert input_vec.shape[0] == len(like_dict)
    i = 0
    out_dict = {}
    for key in like_dict:
        out_dict[key] = input_vec[i]
        i += 1
    return out_dict

def apply_log10(inputs, like_dict):
    i = 0
    for input_name in like_dict:
        if '_log10' in input_name:
            inputs[:, i, :] = 10 ** inputs[:, i, :]
        i += 1

def save_as_cgyro(input, directory):
    try:
        os.mkdir(directory)
        print(f"Directory '{directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory}' already exists.")
    initial_path = os.path.join(directory, 'input.cgyro')
    tglf_path = initial_path[:-6] + '.tglf'
    save_as_tglf(input, tglf_path)
    pyro = Pyro(gk_file=tglf_path, )
    pyro.write_gk_file(initial_path, gk_code='CGYRO', enforce_quasineutrality=False)
    for i in range(len(KY_LOCS)):
        ky = KY_LOCS[i]
        input_dir = os.path.join(directory, f'input-{format_input_num(i)}')
        try:
            os.mkdir(input_dir)
        except FileExistsError:
            print(f"Directory '{input_dir}' already exists.")
        input_path = os.path.join(input_dir, 'input.cgyro')
        overwrite_values(initial_path, input_path, ky)
    os.remove(initial_path)
    os.remove(tglf_path)
    print(f'Successfully saved {directory}')

def save_as_tglf(input, path):
    
    with open(path, "w") as file:
        for key in input:
            if("_log10" in key):
                file.write(f"{key[:-len('_log10')]} = {10 ** input[key]}\n")
            else:
                file.write(f"{key} = {input[key]}\n")
        for key in TGLF_CONSTANTS:
            file.write(f"{key} = {TGLF_CONSTANTS[key]}\n")
        file.close()
    print(f'Successfully saved {path}')

def overwrite_values(path, new_path, ky_value):
    with open(path, 'r') as f:
        lines = f.readlines()
        f.close()
    newlines = []
    for line in lines:
        if 'KY' in line:
            newline = f'KY = {ky_value}\n'
        else:
            foundKey = False
            for key in CGYRO_CONSTANTS:
                if key in line:
                    newline = f'{key} = {CGYRO_CONSTANTS[key]}\n'
                    foundKey = True
                    break
            if not foundKey:
                newline = line
        newlines.append(newline)

    with open(new_path, 'w') as f:
        for line in newlines:
            f.write(line)
        f.close()
    return
            

def save_perturbed_inputs(inputs, directory, like_dict):
    #inputs should have shape [num perturbable params, num input params, 2]
    try:
        os.mkdir(directory)
        print(f"Directory '{directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory}' already exists.")
    
    M = inputs.shape[0]
    for i in range(M):
        for j in range(2):
            input = inputs[i,:,j]
            input_dict = to_dict(input, like_dict)
            input_dir = os.path.join(directory, f'input-{format_input_num((i*2)+j)}')
            input_file = os.path.join(input_dir, 'input.tglf')
            try:
                os.mkdir(input_dir)
                print(f"Directory '{input_dir}' created successfully.")
            except FileExistsError:
                print(f"Directory '{input_dir}' already exists.")
            save_as_tglf(input_dict, input_file)

def format_input_num(num):
    if num < 10:
        return f'00{num}'
    elif num < 100:
        return f'0{num}'
    else:
        return f'{num}'