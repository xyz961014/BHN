#!/bin/bash

# Create a directory to store the output files
output_dir="experiment_output"
mkdir -p ${noise_exeriments_output_dir}

# Set the range of noise rates you want to test
for dataset in cifar-10 cifar-100
do
  for noise_type in symmetric asymmetric instance
  do
    for noise_eta in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        echo "Running experiment with noise_eta=${noise_eta}"

        # Define the output file name based on the noise rate
        output_file="${noise_exeriments_output_dir}/output_${dataset/noise_type/noise_eta}.txt"

        # Run your Python script with the current noise_eta and redirect stdout to the output file
        python train.py dataset=${dataset} noise_type=${noise_type} noise_eta=${noise_eta} > ${output_file} 2>&1

        echo "Experiment complete. Output saved to ${output_file}"

        # Add any additional sleep or commands if needed between experiments
        # sleep 1
    done
  done
done