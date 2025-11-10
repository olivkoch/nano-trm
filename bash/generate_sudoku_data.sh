#!/bin/bash
# Quick script to generate common Sudoku datasets

set -e

echo "=================================="
echo "Sudoku Dataset Generator"
echo "=================================="
echo ""
echo "Select a configuration:"
echo "1) 4x4 - Small (10k train, 1k val) - Quick"
echo "2) 4x4 - Medium (50k train, 5k val)"
echo "3) 4x4 - Large (100k train, 10k val)"
echo "4) 6x6 - Medium (50k train, 5k val)"
echo "5) Custom configuration"
echo ""
read -p "Enter choice [1-5]: " choice

cmd="uv run python scripts/data/generate_sudoku_data.py"

case $choice in
    1)
        echo "Generating 4x4 Small dataset..."
        $cmd \
            --grid-size 4 \
            --num-train 10000 \
            --num-val 1000 \
            --num-test 1000 \
            --output-dir ./data/sudoku_4x4_small
        ;;
    2)
        echo "Generating 4x4 Medium dataset..."
        $cmd \
            --grid-size 4 \
            --num-train 50000 \
            --num-val 5000 \
            --num-test 5000 \
            --output-dir ./data/sudoku_4x4_medium
        ;;
    3)
        echo "Generating 4x4 Large dataset..."
        $cmd \
            --grid-size 4 \
            --num-train 100000 \
            --num-val 10000 \
            --num-test 10000 \
            --output-dir ./data/sudoku_4x4_large
        ;;
    4)
        echo "Generating 6x6 Medium dataset..."
        $cmd \
            --grid-size 6 \
            --num-train 50000 \
            --num-val 5000 \
            --num-test 5000 \
            --output-dir ./data/sudoku_6x6_medium
        ;;
    5)
        read -p "Grid size [4]: " grid_size
        grid_size=${grid_size:-4}
        
        read -p "Train samples [10000]: " num_train
        num_train=${num_train:-10000}
        
        read -p "Val samples [1000]: " num_val
        num_val=${num_val:-1000}
        
        read -p "Test samples [1000]: " num_test
        num_test=${num_test:-1000}
        
        read -p "Output directory [./data/sudoku_custom]: " output_dir
        output_dir=${output_dir:-./data/sudoku_custom}
        
        echo "Generating custom dataset..."
        $cmd \
            --grid-size $grid_size \
            --num-train $num_train \
            --num-val $num_val \
            --num-test $num_test \
            --output-dir $output_dir
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✓ Done! Dataset ready for training."