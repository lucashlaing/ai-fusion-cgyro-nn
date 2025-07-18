import numpy as np
import os
import json

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
            save_input(input_dict, input_file)

def save_input(input, path):
    with open(path, "w") as file:
        for key in input:
            file.write(f"{key} = {input[key]}\n")
        file.close()
    print(f'Successfully saved {path}')

def format_input_num(num):
    if num < 10:
        return f'00{num}'
    elif num < 100:
        return f'0{num}'
    else:
        return f'{num}'