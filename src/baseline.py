"""
ARC-AGI-2 Kaggle Competition Baseline Solution
A single baseline approach that analyzes patterns and applies the most likely transformation
"""

import json
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")


@dataclass
class TransformationPattern:
    """Stores information about a detected transformation pattern."""

    pattern_type: str
    confidence: float
    params: Dict[str, Any]


class ARCBaseline:
    """Baseline solver for ARC-AGI-2 Kaggle competition."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the baseline solver.

        Args:
            data_dir: Directory containing the Kaggle JSON files
        """
        self.data_dir = Path(data_dir)
        self.training_challenges = None
        self.training_solutions = None
        self.evaluation_challenges = None
        self.evaluation_solutions = None

    def load_data(self) -> Dict[str, int]:
        """Load all available data files.

        Returns:
            Dictionary with counts of loaded tasks
        """
        files = {
            "training_challenges": "arc-agi_training_challenges.json",
            "training_solutions": "arc-agi_training_solutions.json",
            "evaluation_challenges": "arc-agi_evaluation_challenges.json",
            "evaluation_solutions": "arc-agi_evaluation_solutions.json",
        }

        loaded = {}
        for key, filename in files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    data = json.load(f)
                    setattr(self, key, data)
                    loaded[key] = len(data)
                    print(f"✓ Loaded {filename}: {len(data)} tasks")
            else:
                print(f"✗ Not found: {filename}")
                loaded[key] = 0

        return loaded

    def analyze_transformation(self, train_examples: List[Dict]) -> TransformationPattern:
        """Analyze training examples to identify the transformation pattern.

        This is the core of the baseline - it tries to identify common patterns.

        Args:
            train_examples: List of training input/output pairs

        Returns:
            TransformationPattern with the most likely pattern
        """
        patterns = []

        # Check 1: Direct copy (output == input)
        if self._check_direct_copy(train_examples):
            patterns.append(TransformationPattern("direct_copy", 1.0, {}))

        # Check 2: Consistent output shape
        output_shape = self._check_consistent_shape(train_examples)
        if output_shape:
            patterns.append(
                TransformationPattern("fixed_output_shape", 0.9, {"shape": output_shape})
            )

        # Check 3: Color mapping/replacement
        color_map = self._check_color_mapping(train_examples)
        if color_map:
            patterns.append(TransformationPattern("color_mapping", 0.8, {"mapping": color_map}))

        # Check 4: Most common color fill
        fill_color = self._check_fill_pattern(train_examples)
        if fill_color is not None:
            patterns.append(TransformationPattern("color_fill", 0.7, {"color": fill_color}))

        # Check 5: Size scaling
        scale_factor = self._check_scaling(train_examples)
        if scale_factor:
            patterns.append(TransformationPattern("scaling", 0.6, {"factor": scale_factor}))

        # Check 6: Pattern extraction (find and replicate a sub-pattern)
        sub_pattern = self._check_pattern_extraction(train_examples)
        if sub_pattern is not None:
            patterns.append(
                TransformationPattern("pattern_extraction", 0.5, {"pattern": sub_pattern})
            )

        # Return the highest confidence pattern, or a default
        if patterns:
            return max(patterns, key=lambda p: p.confidence)
        else:
            # Default: copy input or create small grid
            return TransformationPattern("default", 0.1, {})

    def _check_direct_copy(self, train_examples: List[Dict]) -> bool:
        """Check if outputs are identical to inputs."""
        for example in train_examples:
            if not np.array_equal(example["input"], example["output"]):
                return False
        return len(train_examples) > 0

    def _check_consistent_shape(self, train_examples: List[Dict]) -> Optional[Tuple[int, int]]:
        """Check if all outputs have the same shape."""
        if not train_examples:
            return None

        shapes = [tuple(np.array(ex["output"]).shape) for ex in train_examples]

        # All shapes must be identical
        if len(set(shapes)) == 1:
            return shapes[0]
        return None

    def _check_color_mapping(self, train_examples: List[Dict]) -> Optional[Dict[int, int]]:
        """Check for consistent color transformations."""
        if not train_examples:
            return None

        # Only works if shapes are consistent
        shape_match = all(
            np.array(ex["input"]).shape == np.array(ex["output"]).shape for ex in train_examples
        )

        if not shape_match:
            return None

        # Find consistent color mappings
        all_mappings = []

        for example in train_examples:
            inp = np.array(example["input"])
            out = np.array(example["output"])

            # Get unique color mappings for this example
            mapping = {}
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    in_color = int(inp[i, j])
                    out_color = int(out[i, j])
                    if in_color not in mapping:
                        mapping[in_color] = out_color
                    elif mapping[in_color] != out_color:
                        # Inconsistent mapping within this example
                        return None

            all_mappings.append(mapping)

        # Check if mappings are consistent across all examples
        if not all_mappings:
            return None

        first_mapping = all_mappings[0]
        for mapping in all_mappings[1:]:
            if mapping != first_mapping:
                return None

        return first_mapping

    def _check_fill_pattern(self, train_examples: List[Dict]) -> Optional[int]:
        """Find the most common color in outputs."""
        all_colors = []
        for example in train_examples:
            out = np.array(example["output"])
            all_colors.extend(out.flatten().tolist())

        if all_colors:
            color_counts = Counter(all_colors)
            most_common = color_counts.most_common(1)[0]
            # Only return if this color dominates (>50% of pixels)
            if most_common[1] > len(all_colors) * 0.5:
                return most_common[0]

        return None

    def _check_scaling(self, train_examples: List[Dict]) -> Optional[int]:
        """Check if there's a consistent scaling factor."""
        if not train_examples:
            return None

        scale_factors = []
        for example in train_examples:
            in_shape = np.array(example["input"]).shape
            out_shape = np.array(example["output"]).shape

            # Check if one dimension scales consistently
            if in_shape[0] > 0 and in_shape[1] > 0:
                h_scale = out_shape[0] / in_shape[0]
                w_scale = out_shape[1] / in_shape[1]

                # Both dimensions should scale equally
                if abs(h_scale - w_scale) < 0.01 and h_scale == int(h_scale):
                    scale_factors.append(int(h_scale))

        # Check if all scale factors are the same
        if scale_factors and len(set(scale_factors)) == 1:
            return scale_factors[0]

        return None

    def _check_pattern_extraction(self, train_examples: List[Dict]) -> Optional[np.ndarray]:
        """Check if outputs contain a repeated sub-pattern from inputs."""
        # This is a simplified check - just look for small repeated patterns
        if not train_examples:
            return None

        # Check if all outputs are the same
        outputs = [np.array(ex["output"]) for ex in train_examples]

        if len(outputs) > 1:
            first_output = outputs[0]
            if all(np.array_equal(first_output, out) for out in outputs[1:]):
                # All outputs are identical - might be extracting a pattern
                return first_output

        return None

    def apply_transformation(
        self,
        test_input: List[List[int]],
        pattern: TransformationPattern,
        train_examples: List[Dict],
    ) -> List[List[int]]:
        """Apply the detected transformation pattern to test input.

        Args:
            test_input: The test input grid
            pattern: The detected transformation pattern
            train_examples: Training examples for reference

        Returns:
            Predicted output grid
        """
        input_arr = np.array(test_input)

        if pattern.pattern_type == "direct_copy":
            return test_input

        elif pattern.pattern_type == "fixed_output_shape":
            shape = pattern.params["shape"]
            # Create output of fixed shape, try to fill intelligently
            output = np.zeros(shape, dtype=int)

            # If possible, copy part of input
            min_h = min(shape[0], input_arr.shape[0])
            min_w = min(shape[1], input_arr.shape[1])
            output[:min_h, :min_w] = input_arr[:min_h, :min_w]

            return output.tolist()

        elif pattern.pattern_type == "color_mapping":
            mapping = pattern.params["mapping"]
            output = input_arr.copy()

            for old_color, new_color in mapping.items():
                output[input_arr == old_color] = new_color

            return output.tolist()

        elif pattern.pattern_type == "color_fill":
            fill_color = pattern.params["color"]
            # Determine output shape from training examples
            if train_examples:
                out_shape = np.array(train_examples[0]["output"]).shape
                output = np.full(out_shape, fill_color, dtype=int)
            else:
                output = np.full_like(input_arr, fill_color)

            return output.tolist()

        elif pattern.pattern_type == "scaling":
            factor = pattern.params["factor"]
            output = np.repeat(np.repeat(input_arr, factor, axis=0), factor, axis=1)
            return output.tolist()

        elif pattern.pattern_type == "pattern_extraction":
            # Return the extracted pattern
            extracted = pattern.params["pattern"]
            return extracted.tolist()

        else:  # default
            # Default strategy: try to match output shape from examples
            if train_examples and train_examples[0]["output"]:
                target_shape = np.array(train_examples[0]["output"]).shape

                # If shapes match, copy input
                if input_arr.shape == target_shape:
                    return test_input
                else:
                    # Create output of target shape
                    output = np.zeros(target_shape, dtype=int)
                    min_h = min(target_shape[0], input_arr.shape[0])
                    min_w = min(target_shape[1], input_arr.shape[1])
                    output[:min_h, :min_w] = input_arr[:min_h, :min_w]
                    return output.tolist()
            else:
                return test_input

    def solve_task(self, task_data: Dict) -> List[List[List[int]]]:
        """Solve a single task with all its test cases.

        Args:
            task_data: Dictionary containing 'train' and 'test' examples

        Returns:
            List of predicted outputs for each test case
        """
        train_examples = task_data.get("train", [])
        test_examples = task_data.get("test", [])

        # Analyze the transformation pattern
        pattern = self.analyze_transformation(train_examples)

        # Apply transformation to each test case
        predictions = []
        for test_case in test_examples:
            test_input = test_case["input"]
            prediction = self.apply_transformation(test_input, pattern, train_examples)
            predictions.append(prediction)

        return predictions

    def create_submission(
        self, test_file: str = "arc-agi_test_challenges.json", output_file: str = "submission.json"
    ) -> Dict:
        """Create a submission file for Kaggle competition.

        Args:
            test_file: Name of the test challenges file
            output_file: Name of the output submission file

        Returns:
            Submission dictionary
        """
        test_path = self.data_dir / test_file

        if not test_path.exists():
            print(f"Error: Test file {test_file} not found!")
            print("For local testing, you can use evaluation challenges.")
            return {}

        # Load test challenges
        with open(test_path) as f:
            test_challenges = json.load(f)

        print(f"Creating submission for {len(test_challenges)} tasks...")

        # Create predictions for each task
        submission = {}

        for task_id, task_data in test_challenges.items():
            predictions = self.solve_task(task_data)

            # Format for Kaggle: list of output grids
            submission[task_id] = predictions

        # Save submission
        output_path = self.data_dir / output_file
        with open(output_path, "w") as f:
            json.dump(submission, f)

        print(f"✓ Submission saved to {output_path}")
        print(f"  Total tasks: {len(submission)}")

        return submission

    def evaluate(self, dataset: str = "training", max_tasks: int = None) -> Dict[str, float]:
        """Evaluate the baseline on a dataset with solutions.

        Args:
            dataset: 'training' or 'evaluation'
            max_tasks: Maximum number of tasks to evaluate (None for all)

        Returns:
            Dictionary with evaluation metrics
        """
        if dataset == "training":
            challenges = self.training_challenges
            solutions = self.training_solutions
        else:
            challenges = self.evaluation_challenges
            solutions = self.evaluation_solutions

        if not challenges or not solutions:
            print(f"Error: {dataset} data not loaded!")
            return {}

        task_ids = list(challenges.keys())
        if max_tasks:
            task_ids = task_ids[:max_tasks]

        print(f"Evaluating on {len(task_ids)} {dataset} tasks...")

        correct = 0
        total = 0
        task_accuracies = []

        for task_id in task_ids:
            task_data = challenges[task_id]
            true_outputs = solutions[task_id]

            # Get predictions
            predictions = self.solve_task(task_data)

            # Compare with ground truth
            task_correct = 0
            task_total = len(predictions)

            for i, pred in enumerate(predictions):
                if i < len(true_outputs):
                    # Handle different solution formats
                    if isinstance(true_outputs[i], dict):
                        true_output = true_outputs[i]["output"]
                    else:
                        true_output = true_outputs[i]

                    if np.array_equal(pred, true_output):
                        task_correct += 1
                        correct += 1

                    total += 1

            if task_total > 0:
                task_acc = task_correct / task_total
                task_accuracies.append(task_acc)

        # Calculate metrics
        overall_accuracy = correct / total if total > 0 else 0
        task_solve_rate = (
            sum(1 for acc in task_accuracies if acc == 1.0) / len(task_accuracies)
            if task_accuracies
            else 0
        )

        metrics = {
            "overall_accuracy": overall_accuracy,
            "task_solve_rate": task_solve_rate,
            "correct_outputs": correct,
            "total_outputs": total,
            "tasks_evaluated": len(task_ids),
        }

        print("\n📊 Evaluation Results:")
        print(f"  Overall accuracy: {overall_accuracy:.2%}")
        print(f"  Task solve rate: {task_solve_rate:.2%}")
        print(f"  Correct outputs: {correct}/{total}")

        return metrics

    def analyze_patterns(self, dataset: str = "training", sample_size: int = 50) -> None:
        """Analyze and print pattern distribution in the dataset.

        Args:
            dataset: Which dataset to analyze
            sample_size: Number of tasks to sample
        """
        if dataset == "training":
            challenges = self.training_challenges
        else:
            challenges = self.evaluation_challenges

        if not challenges:
            print(f"No {dataset} data loaded!")
            return

        task_ids = list(challenges.keys())[:sample_size]
        pattern_counts = Counter()

        print(f"\nAnalyzing patterns in {len(task_ids)} {dataset} tasks...")

        for task_id in task_ids:
            task_data = challenges[task_id]
            train_examples = task_data.get("train", [])

            if train_examples:
                pattern = self.analyze_transformation(train_examples)
                pattern_counts[pattern.pattern_type] += 1

        print("\n📊 Pattern Distribution:")
        for pattern_type, count in pattern_counts.most_common():
            percentage = count / len(task_ids) * 100
            print(f"  {pattern_type}: {count} ({percentage:.1f}%)")


def main():
    """Main function to run the baseline."""
    print("🧩 ARC-AGI-2 Kaggle Baseline Solution")
    print("=" * 50)

    # Initialize baseline
    baseline = ARCBaseline(data_dir="data")

    # Load data
    print("\nLoading data...")
    loaded = baseline.load_data()

    if not any(loaded.values()):
        print("\n❌ No data found!")
        print("Please ensure your Kaggle data files are in the 'data' directory:")
        print("  - arc-agi_training_challenges.json")
        print("  - arc-agi_training_solutions.json")
        print("  - arc-agi_evaluation_challenges.json")
        print("  - arc-agi_evaluation_solutions.json")
        return

    # Analyze pattern distribution
    if baseline.training_challenges:
        baseline.analyze_patterns("training", sample_size=100)

    # Evaluate on training set
    if baseline.training_challenges and baseline.training_solutions:
        print("\n" + "=" * 50)
        print("Evaluating on Training Set")
        print("=" * 50)
        baseline.evaluate("training", max_tasks=50)

    # Evaluate on evaluation set
    if baseline.evaluation_challenges and baseline.evaluation_solutions:
        print("\n" + "=" * 50)
        print("Evaluating on Evaluation Set")
        print("=" * 50)
        baseline.evaluate("evaluation", max_tasks=30)

    # Create submission (if test file exists)
    print("\n" + "=" * 50)
    print("Creating Submission")
    print("=" * 50)

    test_path = baseline.data_dir / "arc-agi_test_challenges.json"
    if test_path.exists():
        baseline.create_submission()
    else:
        print("ℹ️  Test file not found. For Kaggle competition, this will be provided.")
        print("   You can test submission creation with evaluation challenges:")
        baseline.create_submission(
            test_file="arc-agi_evaluation_challenges.json", output_file="test_submission.json"
        )

    print("\n✅ Baseline analysis complete!")
    print("\nNext steps to improve:")
    print("  1. Add rotation/reflection detection")
    print("  2. Implement flood fill and object detection")
    print("  3. Add symmetry and repetition patterns")
    print("  4. Implement more sophisticated pattern matching")
    print("  5. Consider ensemble methods combining multiple strategies")


if __name__ == "__main__":
    main()
