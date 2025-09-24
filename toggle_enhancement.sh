#!/bin/bash

# Script to toggle between original and enhanced versions

if [ "$1" = "enhanced" ]; then
    echo "Switching to enhanced version with post-processing..."
    cp datasets/humaneval_dataset_enhanced.py datasets/humaneval_dataset.py
    echo "✅ Enhanced version active (with post-processing)"
    echo "Run your experiment to see improved results"
elif [ "$1" = "original" ]; then
    echo "Switching to original version..."
    cp datasets/humaneval_dataset.py.backup4 datasets/humaneval_dataset.py
    echo "✅ Original version active (no post-processing)"
    echo "Run your experiment to see original results"
else
    echo "Usage: $0 [enhanced|original]"
    echo ""
    echo "enhanced  - Use enhanced version with post-processing"
    echo "original  - Use original version without post-processing"
    echo ""
    echo "Current version:"
    if grep -q "post_process_response" datasets/humaneval_dataset.py; then
        echo "✅ Enhanced version (with post-processing)"
    else
        echo "✅ Original version (no post-processing)"
    fi
fi
