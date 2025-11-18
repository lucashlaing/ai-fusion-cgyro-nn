#!/bin/bash
# $1 : (train_path) s3 path to train data from past iterations
# $2 : (test_path) s3 path to test data (should not change between iterations)
# $3 : (tglf_candidates_path) s3 path to formatted tglf candidate inputs, ready for simulation
# $4 : (candidates_h5_path) s3 path to candidate h5 file for logging purposes
# $5 : (last_model_checkpoint_path) s3 path to last model, for use in training
# $6 : (new_model_checkpoint_path) s3 path to save model to after training complete
cp ./config/launch_template.yaml ./config/launch.yaml
echo "dataset:
  default:
    hparam:
      _train_path: $1
      _test_path: $2
      _candidates_tglf_path: $3
      _candidates_h5_path: $4
      _last_model_checkpoint_path: $5
      _new_model_checkpoint_path: $6
run:
  model: [tglf-online-bal]
  dataset: [default]" >> ./config/launch.yaml
echo "Configured launch.yaml, making train job"
make job