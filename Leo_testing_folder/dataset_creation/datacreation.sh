#!/bin/bash

# Specify the base paths
input_base="data/asl_alphabet_train_wo_test/asl_alphabet_train"
output_base="Leo_testing_folder/classifier_csvs/training_csvs"

# Loop over the alphabet
for letter in {A..Z}; do
    echo "Running Command for letter $letter"
    
    # Construct the paths for this letter
    input_path="$input_base/$letter"
    output_path="$output_base/$letter"
    
    # Run spark-submit
    spark-submit folder_to_labels.py "$input_path" "$output_path"
done

echo "All commands executed successfully"