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
echo "    2) 4x4 - Medium (6k base × 8 = 48k train) - Fast"
echo "    3) 4x4 - Large (12k base × 8 = 96k train)"
echo ""
echo "  6x6 Datasets:"
echo "    4) 6x6 - Small (5k train, 1k val) - Quick"
echo "    5) 6x6 - Medium (6k base × 8 = 48k train) - Fast"
echo "    6) 6x6 - Large (12k base × 8 = 96k train)"
echo "    7) 6x6 - Hybrid (5k train std, 1k val/test OOD)"
echo ""
echo "  9x9 Datasets:"
echo "    8) 9x9 - Small (5k train, 1k val) - Quick"
echo "    9) 9x9 - Medium (6k base × 8 = 48k train) - Fast"
echo "   10) 9x9 - Large (12k base × 8 = 96k train)"
echo "   11) 9x9 - Hybrid (5k train std, 1k val/test OOD)"
echo ""
echo "  Multi-size:"
echo "   12) Mixed 6x6+9x9 (6k base × 8 = 48k train)"
echo "   13) Cross-size 6x6→9x9 (6k base × 8 = 48k train)"
echo ""
echo "  Other:"
echo "    c) Custom configuration"
echo ""
read -p "Enter choice [1-13, c]: " choice

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
        echo "Generating 4x4 Medium dataset (6k × 8 = 48k train)..."
        $cmd \
            --grid-size 4 \
            --num-train 6000 \
            --num-val 1000 \
            --num-test 1000 \
            --num-aug 7 \
            --output-dir ./data/sudoku_4x4_medium
        ;;
    3)
        echo "Generating 4x4 Large dataset (12k × 8 = 96k train)..."
        $cmd \
            --grid-size 4 \
            --num-train 12000 \
            --num-val 2000 \
            --num-test 2000 \
            --num-aug 7 \
            --output-dir ./data/sudoku_4x4_large
        ;;
    4)
        echo "Generating 6x6 Small dataset..."
        $cmd \
            --grid-size 6 \
            --num-train 5000 \
            --num-val 1000 \
            --num-test 1000 \
            --full \
            --output-dir ./data/sudoku_6x6_small
        ;;
    5)
        echo "Generating 6x6 Medium dataset (6k × 8 = 48k train)..."
        $cmd \
            --grid-size 6 \
            --num-train 6000 \
            --num-val 1000 \
            --num-test 1000 \
            --num-aug 7 \
            --full \
            --output-dir ./data/sudoku_6x6_medium
        ;;
    6)
        echo "Generating 6x6 Large dataset (12k × 8 = 96k train)..."
        $cmd \
            --grid-size 6 \
            --num-train 12000 \
            --num-val 2000 \
            --num-test 2000 \
            --num-aug 7 \
            --full \
            --output-dir ./data/sudoku_6x6_large
        ;;
    7)
        echo "Generating 6x6 Hybrid dataset (OOD test set)..."
        $cmd \
            --grid-size 6 \
            --num-train 5000 \
            --num-val 1000 \
            --num-test 1000 \
            --hybrid \
            --output-dir ./data/sudoku_6x6_hybrid
        ;;
    8)
        echo "Generating 9x9 Small dataset..."
        $cmd \
            --grid-size 9 \
            --num-train 5000 \
            --num-val 1000 \
            --num-test 1000 \
            --full \
            --output-dir ./data/sudoku_9x9_small
        ;;
    9)
        echo "Generating 9x9 Medium dataset (6k × 8 = 48k train)..."
        $cmd \
            --grid-size 9 \
            --num-train 6000 \
            --num-val 1000 \
            --num-test 1000 \
            --num-aug 7 \
            --full \
            --output-dir ./data/sudoku_9x9_medium
        ;;
    10)
        echo "Generating 9x9 Large dataset (12k × 8 = 96k train)..."
        $cmd \
            --grid-size 9 \
            --num-train 12000 \
            --num-val 2000 \
            --num-test 2000 \
            --num-aug 7 \
            --full \
            --output-dir ./data/sudoku_9x9_large
        ;;
    11)
        echo "Generating 9x9 Hybrid dataset (OOD test set)..."
        $cmd \
            --grid-size 9 \
            --num-train 5000 \
            --num-val 1000 \
            --num-test 1000 \
            --hybrid \
            --output-dir ./data/sudoku_9x9_hybrid
        ;;
    12)
        echo "Generating Mixed-size dataset (6k × 8 = 48k train)..."
        $cmd \
            --num-train 6000 \
            --num-val 1000 \
            --num-test 1000 \
            --num-aug 7 \
            --mixed-size \
            --output-dir ./data/sudoku_mixed_6x6_9x9
        ;;
    13)
        echo "Generating Cross-size dataset (6k × 8 = 48k train on 6x6, val/test on 9x9)..."
        $cmd \
            --num-train 6000 \
            --num-val 1000 \
            --num-test 1000 \
            --num-aug 7 \
            --cross-size \
            --output-dir ./data/sudoku_cross_6x6_9x9
        ;;
    c|C)
        echo "Select mode:"
        echo "  1) Single grid size (standard/full/hybrid)"
        echo "  2) Mixed-size (50% 6x6, 50% 9x9 in all splits)"
        echo "  3) Cross-size (train: 6x6, val/test: 9x9)"
        read -p "Mode type [1-3] (default: 1): " mode_type
        mode_type=${mode_type:-1}
        
        if [ "$mode_type" == "2" ]; then
            read -p "Train base puzzles [6000]: " num_train
            num_train=${num_train:-6000}
            
            read -p "Val samples [1000]: " num_val
            num_val=${num_val:-1000}
            
            read -p "Test samples [1000]: " num_test
            num_test=${num_test:-1000}
            
            read -p "Augmentations per train puzzle [0-7] (default: 7): " num_aug
            num_aug=${num_aug:-7}
            
            read -p "Output directory [./data/sudoku_mixed_custom]: " output_dir
            output_dir=${output_dir:-./data/sudoku_mixed_custom}
            
            aug_flag=""
            if [ "$num_aug" != "0" ]; then
                aug_flag="--num-aug $num_aug"
            fi
            
            echo "Generating mixed-size dataset (50% 6x6, 50% 9x9)..."
            $cmd \
                --num-train $num_train \
                --num-val $num_val \
                --num-test $num_test \
                --mixed-size \
                $aug_flag \
                --output-dir $output_dir
                
        elif [ "$mode_type" == "3" ]; then
            read -p "Train base puzzles (6x6) [6000]: " num_train
            num_train=${num_train:-6000}
            
            read -p "Val samples (9x9) [1000]: " num_val
            num_val=${num_val:-1000}
            
            read -p "Test samples (9x9) [1000]: " num_test
            num_test=${num_test:-1000}
            
            read -p "Augmentations per train puzzle [0-7] (default: 7): " num_aug
            num_aug=${num_aug:-7}
            
            read -p "Output directory [./data/sudoku_cross_custom]: " output_dir
            output_dir=${output_dir:-./data/sudoku_cross_custom}
            
            aug_flag=""
            if [ "$num_aug" != "0" ]; then
                aug_flag="--num-aug $num_aug"
            fi
            
            echo "Generating cross-size dataset (train: 6x6, val/test: 9x9)..."
            $cmd \
                --num-train $num_train \
                --num-val $num_val \
                --num-test $num_test \
                --cross-size \
                $aug_flag \
                --output-dir $output_dir
        else
            read -p "Grid size [4/6/9] (default: 9): " grid_size
            grid_size=${grid_size:-9}
            
            read -p "Train base puzzles [6000]: " num_train
            num_train=${num_train:-6000}
            
            read -p "Val samples [1000]: " num_val
            num_val=${num_val:-1000}
            
            read -p "Test samples [1000]: " num_test
            num_test=${num_test:-1000}
            
            read -p "Augmentations per train puzzle [0-7] (default: 7): " num_aug
            num_aug=${num_aug:-7}
            
            read -p "Output directory [./data/sudoku_custom]: " output_dir
            output_dir=${output_dir:-./data/sudoku_custom}
            
            mode_flag="--full"
            if [ "$grid_size" == "6" ] || [ "$grid_size" == "9" ]; then
                echo "Select generation mode:"
                echo "  1) Standard (single base grid)"
                echo "  2) Full (multiple base grids, better coverage) [recommended]"
                echo "  3) Hybrid (standard train, full val/test for OOD testing)"
                read -p "Mode [1-3] (default: 2): " mode_choice
                mode_choice=${mode_choice:-2}
                
                case $mode_choice in
                    1) mode_flag="" ;;
                    3) mode_flag="--hybrid" ;;
                    *) mode_flag="--full" ;;
                esac
            fi
            
            aug_flag=""
            if [ "$num_aug" != "0" ]; then
                aug_flag="--num-aug $num_aug"
            fi
            
            echo "Generating custom dataset..."
            $cmd \
                --grid-size $grid_size \
                --num-train $num_train \
                --num-val $num_val \
                --num-test $num_test \
                $mode_flag \
                $aug_flag \
                --output-dir $output_dir
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✓ Done! Dataset ready for training."
echo ""
echo "Tips:"
echo "  - Use num_train_groups (not num_train) for steps/epoch calculations"
echo "  - Augmented puzzles share the same base solution within a group"
echo "  - Run 'python tests/src/nn/data/test_sudoku_data.py <dir> --show-groups' to inspect"