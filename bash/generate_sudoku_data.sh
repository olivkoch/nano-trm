#!/bin/bash
# Quick script to generate common Sudoku datasets
set -e

echo "=================================="
echo "Sudoku Dataset Generator"
echo "=================================="
echo ""
echo "Select a configuration:"
echo ""
echo "  4x4 Datasets:"
echo "    1) 4x4 - Small (10k train, 1k val) - Quick"
echo "    2) 4x4 - Medium (50k train, 5k val)"
echo "    3) 4x4 - Large (100k train, 10k val)"
echo ""
echo "  6x6 Datasets:"
echo "    4) 6x6 - Small (5k train, 1k val)"
echo "    5) 6x6 - Medium (50k train, 5k val)"
echo "    6) 6x6 - Full enumeration (50k train, 5k val) - Uniform sampling"
echo ""
echo "  9x9 Datasets:"
echo "    7) 9x9 - Small (5k train, 1k val)"
echo "    8) 9x9 - Medium (50k train, 5k val)"
echo "    9) 9x9 - Large (100k train, 10k val)"
echo ""
echo "  Other:"
echo "    c) Custom configuration"
echo ""
read -p "Enter choice [1-9, c]: " choice

cmd="uv run python scripts/data/generate_sudoku_data.py"

case $choice in
    1)
        echo "Generating 4x4 Small dataset..."
        $cmd \
            --grid-size 4 \
            --max-grid-size 9 \
            --num-train 10000 \
            --num-val 1000 \
            --num-test 1000 \
            --output-dir ./data/sudoku_4x4_small
        ;;
    2)
        echo "Generating 4x4 Medium dataset..."
        $cmd \
            --grid-size 4 \
            --max-grid-size 9 \
            --num-train 50000 \
            --num-val 5000 \
            --num-test 5000 \
            --output-dir ./data/sudoku_4x4_medium
        ;;
    3)
        echo "Generating 4x4 Large dataset..."
        $cmd \
            --grid-size 4 \
            --max-grid-size 9 \
            --num-train 100000 \
            --num-val 10000 \
            --num-test 10000 \
            --output-dir ./data/sudoku_4x4_large
        ;;
    4)
        echo "Generating 6x6 Small dataset..."
        $cmd \
            --grid-size 6 \
            --max-grid-size 9 \
            --num-train 5000 \
            --num-val 1000 \
            --num-test 1000 \
            --output-dir ./data/sudoku_6x6_small
        ;;
    5)
        echo "Generating 6x6 Medium dataset..."
        $cmd \
            --grid-size 6 \
            --max-grid-size 9 \
            --num-train 50000 \
            --num-val 5000 \
            --num-test 5000 \
            --output-dir ./data/sudoku_6x6_medium
        ;;
    6)
        echo "Generating 6x6 Full enumeration dataset..."
        echo "(First run will enumerate all 28M grids and cache base grids)"
        $cmd \
            --grid-size 6 \
            --max-grid-size 9 \
            --num-train 50000 \
            --num-val 5000 \
            --num-test 5000 \
            --full \
            --output-dir ./data/sudoku_6x6_medium_full
        ;;
    7)
        echo "Generating 9x9 Small dataset..."
        $cmd \
            --grid-size 9 \
            --max-grid-size 9 \
            --num-train 5000 \
            --num-val 1000 \
            --num-test 1000 \
            --output-dir ./data/sudoku_9x9_small
        ;;
    8)
        echo "Generating 9x9 Medium dataset..."
        $cmd \
            --grid-size 9 \
            --max-grid-size 9 \
            --num-train 50000 \
            --num-val 5000 \
            --num-test 5000 \
            --output-dir ./data/sudoku_9x9_medium
        ;;
    9)
        echo "Generating 9x9 Large dataset..."
        $cmd \
            --grid-size 9 \
            --max-grid-size 9 \
            --num-train 100000 \
            --num-val 10000 \
            --num-test 10000 \
            --output-dir ./data/sudoku_9x9_large
        ;;
    c|C)
        read -p "Grid size [4/6/9] (default: 4): " grid_size
        grid_size=${grid_size:-4}
        
        read -p "Train samples [10000]: " num_train
        num_train=${num_train:-10000}
        
        read -p "Val samples [1000]: " num_val
        num_val=${num_val:-1000}
        
        read -p "Test samples [1000]: " num_test
        num_test=${num_test:-1000}
        
        read -p "Output directory [./data/sudoku_custom]: " output_dir
        output_dir=${output_dir:-./data/sudoku_custom}
        
        if [ "$grid_size" == "6" ]; then
            read -p "Use full enumeration? [y/N]: " use_full
            if [ "$use_full" == "y" ] || [ "$use_full" == "Y" ]; then
                full_flag="--full"
            else
                full_flag=""
            fi
        else
            full_flag=""
        fi
        
        echo "Generating custom dataset..."
        $cmd \
            --grid-size $grid_size \
            --num-train $num_train \
            --num-val $num_val \
            --num-test $num_test \
            $full_flag \
            --output-dir $output_dir
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✓ Done! Dataset ready for training."