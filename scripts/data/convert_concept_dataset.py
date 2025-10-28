"""
Convert external ARC-like dataset to standard ARC format.

Directory structure:
corpus/
  ├── Count/
  │   ├── task1.json
  │   ├── task2.json
  │   └── ...
  ├── Copy/
  │   ├── task1.json
  │   └── ...
  └── ...

Output:
  ├── concept_training_challenges.json
  ├── concept_training_solutions.json
  ├── concept_evaluation_challenges.json
  └── concept_evaluation_solutions.json
"""

import hashlib
import json
import random
import string
from pathlib import Path
from typing import Dict, List, Set, Tuple

import click


def generate_unique_task_id(
    concept: str, filename: str, used_ids: Set[str], max_attempts: int = 1000
) -> str:
    """
    Generate a unique task ID in the format: concept_xxxxxxxx
    Guarantees uniqueness by checking against used_ids set.

    Args:
        concept: Concept name (e.g., "count", "copy")
        filename: Original filename for uniqueness
        used_ids: Set of already used task IDs
        max_attempts: Maximum attempts to find unique ID

    Returns:
        Unique task ID like "count_a3f9c2b1"

    Raises:
        RuntimeError: If unable to generate unique ID after max_attempts
    """
    # Normalize concept name (lowercase, replace spaces/special chars)
    concept_normalized = concept.lower().replace(" ", "_").replace("-", "_")
    concept_normalized = "".join(c for c in concept_normalized if c.isalnum() or c == "_")

    # Truncate concept name if too long (keep first 20 chars)
    if len(concept_normalized) > 20:
        concept_normalized = concept_normalized[:20]

    # Try with MD5 hash first
    hash_input = f"{concept}_{filename}".encode()
    hash_hex = hashlib.md5(hash_input).hexdigest()[:8]
    task_id = f"{concept_normalized}_{hash_hex}"

    if task_id not in used_ids:
        used_ids.add(task_id)
        return task_id

    # If collision, try with additional entropy
    for attempt in range(max_attempts):
        # Add attempt number to hash input
        hash_input = f"{concept}_{filename}_{attempt}".encode()
        hash_hex = hashlib.md5(hash_input).hexdigest()[:8]
        task_id = f"{concept_normalized}_{hash_hex}"

        if task_id not in used_ids:
            used_ids.add(task_id)
            return task_id

    # Last resort: use random characters
    for _ in range(max_attempts):
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        task_id = f"{concept_normalized}_{random_suffix}"

        if task_id not in used_ids:
            used_ids.add(task_id)
            return task_id

    raise RuntimeError(f"Failed to generate unique task ID after {max_attempts * 2} attempts")


def convert_task_format(task_data: Dict, task_id: str) -> Tuple[Dict, List]:
    """
    Convert a single task from external format to ARC format.

    Args:
        task_data: Task data with 'train' and 'test' keys
        task_id: Unique task identifier

    Returns:
        Tuple of (challenge_dict, solution_list)
    """
    # Challenge: contains train examples (with input+output) and test inputs (only input)
    challenge = {
        "train": task_data.get("train", []),  # Keep train as-is (has input and output)
        "test": [],
    }

    # Solution: list of test outputs (just the arrays, not dicts)
    solution = []

    # Process test examples
    for test_example in task_data.get("test", []):
        # Add ONLY input to challenge (no output key)
        challenge["test"].append({"input": test_example["input"]})

        # Add ONLY output array to solution (not wrapped in dict)
        solution.append(test_example["output"])

    return challenge, solution


def load_tasks_from_directory(corpus_dir: Path) -> Dict[str, Dict]:
    """
    Load all tasks from corpus directory structure with unique task IDs.

    Args:
        corpus_dir: Path to corpus directory

    Returns:
        Dictionary mapping task_id -> task_data
    """
    tasks = {}
    used_ids: Set[str] = set()  # Track used IDs to ensure uniqueness
    total_files = 0
    failed_files = 0

    # Iterate through concept directories
    for concept_dir in sorted(corpus_dir.iterdir()):
        if not concept_dir.is_dir():
            continue

        concept_name = concept_dir.name
        print(f"Processing concept: {concept_name}")

        # Iterate through JSON files in concept directory
        json_files = sorted(concept_dir.glob("*.json"))
        print(f"  Found {len(json_files)} tasks")

        for json_file in json_files:
            total_files += 1
            try:
                with open(json_file) as f:
                    task_data = json.load(f)

                # Validate task data structure
                if "train" not in task_data or "test" not in task_data:
                    print(f"  ⚠️  Skipping {json_file.name}: missing 'train' or 'test' key")
                    failed_files += 1
                    continue

                # Generate unique task ID
                task_id = generate_unique_task_id("concept", json_file.name, used_ids)

                # Double-check uniqueness (should never fail due to generate_unique_task_id)
                if task_id in tasks:
                    print(f"  ⚠️  ERROR: Duplicate task ID {task_id}! This should never happen.")
                    failed_files += 1
                    continue

                # Store task data with metadata
                tasks[task_id] = {
                    "data": task_data,
                    "concept": concept_name,
                    "original_file": str(json_file),
                }

            except json.JSONDecodeError as e:
                print(f"  ⚠️  Invalid JSON in {json_file.name}: {e}")
                failed_files += 1
                continue
            except Exception as e:
                print(f"  ⚠️  Error loading {json_file.name}: {e}")
                failed_files += 1
                continue

        print(
            f"  ✓ Loaded {len([t for t in tasks.values() if t['concept'] == concept_name])} tasks from {concept_name}"
        )

    # Verify uniqueness
    assert len(tasks) == len(used_ids), "Task count mismatch with unique IDs!"

    print("\n📊 Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Successfully loaded: {len(tasks)}")
    print(f"  Failed: {failed_files}")
    print(f"  Unique task IDs: {len(used_ids)}")

    return tasks


def split_tasks(
    tasks: Dict[str, Dict], train_ratio: float = 0.8, seed: int = 42
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Split tasks into training and evaluation sets.
    Maintains concept distribution if possible.

    Args:
        tasks: Dictionary of all tasks
        train_ratio: Proportion of tasks for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (training_tasks, evaluation_tasks)
    """
    random.seed(seed)

    # Group tasks by concept
    tasks_by_concept = {}
    for task_id, task_info in tasks.items():
        concept = task_info["concept"]
        if concept not in tasks_by_concept:
            tasks_by_concept[concept] = []
        tasks_by_concept[concept].append(task_id)

    train_tasks = {}
    eval_tasks = {}

    # Split each concept separately to maintain distribution
    for _, task_ids in tasks_by_concept.items():
        # Shuffle task IDs for this concept
        shuffled_ids = list(task_ids)
        random.shuffle(shuffled_ids)

        # Split
        split_idx = max(1, int(len(shuffled_ids) * train_ratio))  # At least 1 in train

        train_ids = shuffled_ids[:split_idx]
        eval_ids = shuffled_ids[split_idx:]

        # Add to train/eval dicts
        for tid in train_ids:
            train_tasks[tid] = tasks[tid]
        for tid in eval_ids:
            eval_tasks[tid] = tasks[tid]

    return train_tasks, eval_tasks


def create_arc_format_files(
    tasks: Dict[str, Dict], output_dir: Path, prefix: str = "concept_training"
) -> Tuple[Path, Path]:
    """
    Create ARC-format JSON files from tasks.

    Args:
        tasks: Dictionary of tasks
        output_dir: Output directory
        prefix: File prefix (e.g., "concept_training" or "concept_evaluation")

    Returns:
        Tuple of (challenges_path, solutions_path)
    """
    challenges = {}
    solutions = {}

    for task_id, task_info in tasks.items():
        task_data = task_info["data"]

        # Convert to ARC format
        challenge, solution = convert_task_format(task_data, task_id)

        challenges[task_id] = challenge
        solutions[task_id] = solution

    # Verify no duplicate keys (should be impossible but extra safety)
    assert len(challenges) == len(tasks), "Duplicate task IDs in challenges!"
    assert len(solutions) == len(tasks), "Duplicate task IDs in solutions!"

    # Save files
    challenges_path = output_dir / f"{prefix}_challenges.json"
    solutions_path = output_dir / f"{prefix}_solutions.json"

    with open(challenges_path, "w") as f:
        json.dump(challenges, f)

    with open(solutions_path, "w") as f:
        json.dump(solutions, f)

    print(f"  ✓ {len(challenges)} unique tasks written")

    return challenges_path, solutions_path


def print_statistics(tasks: Dict[str, Dict], dataset_name: str):
    """Print statistics about the dataset."""
    print(f"\n{'=' * 60}")
    print(f"📊 {dataset_name} Statistics")
    print(f"{'=' * 60}")
    print(f"Total tasks: {len(tasks)}")

    # Verify uniqueness
    task_ids = list(tasks.keys())
    unique_ids = set(task_ids)
    if len(task_ids) != len(unique_ids):
        print("⚠️  WARNING: Found duplicate task IDs!")
        print(f"  Total: {len(task_ids)}, Unique: {len(unique_ids)}")
    else:
        print("✓ All task IDs are unique")

    # Count by concept
    concept_counts = {}
    for task_info in tasks.values():
        concept = task_info["concept"]
        concept_counts[concept] = concept_counts.get(concept, 0) + 1

    print("\nTasks by concept:")
    for concept, count in sorted(concept_counts.items()):
        print(f"  {concept}: {count}")

    # Analyze task sizes
    total_train_examples = 0
    total_test_examples = 0

    for task_info in tasks.values():
        task_data = task_info["data"]
        total_train_examples += len(task_data.get("train", []))
        total_test_examples += len(task_data.get("test", []))

    print("\nAverage examples per task:")
    print(f"  Training: {total_train_examples / len(tasks):.1f}")
    print(f"  Test: {total_test_examples / len(tasks):.1f}")


def verify_no_duplicates(tasks: Dict[str, Dict]) -> bool:
    """
    Verify that all task IDs are unique.

    Returns:
        True if all IDs are unique, False otherwise
    """
    task_ids = list(tasks.keys())
    unique_ids = set(task_ids)

    if len(task_ids) != len(unique_ids):
        # Find duplicates
        from collections import Counter

        id_counts = Counter(task_ids)
        duplicates = [task_id for task_id, count in id_counts.items() if count > 1]

        print(f"\n❌ ERROR: Found {len(duplicates)} duplicate task IDs:")
        for dup_id in duplicates[:10]:  # Show first 10
            print(f"  - {dup_id} (appears {id_counts[dup_id]} times)")

        return False

    return True


@click.command()
@click.option("--corpus-dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="data",
    help="Output directory for ARC-format files",
)
@click.option("--train-ratio", type=float, default=0.8, help="Proportion of tasks for training set")
@click.option("--seed", type=int, default=42, help="Random seed for train/eval split")
@click.option("--no-split", is_flag=True, help="Do not split into train/eval (put all in training)")
@click.option("--verify-only", is_flag=True, help="Only verify uniqueness without creating files")
def main(corpus_dir, output_dir, train_ratio, seed, no_split, verify_only):
    """Main conversion script."""

    # Setup paths
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)

    if not verify_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Converting dataset from: {corpus_dir}")
    if not verify_only:
        print(f"📁 Output directory: {output_dir}")
    print()

    # Load all tasks
    print("Loading tasks...")
    all_tasks = load_tasks_from_directory(corpus_dir)
    print(f"\n✓ Loaded {len(all_tasks)} total tasks")

    # Verify uniqueness
    print("\n🔍 Verifying task ID uniqueness...")
    if verify_no_duplicates(all_tasks):
        print("✓ All task IDs are unique!")
    else:
        print("❌ Duplicate task IDs found! Please check the output above.")
        return 1

    if verify_only:
        print("\n✅ Verification complete!")
        return 0

    if no_split:
        # Put everything in training
        train_tasks = all_tasks
        eval_tasks = {}
        print("\n📝 Mode: All tasks → training set")
    else:
        # Split into train/eval
        print(f"\n📝 Splitting: {train_ratio:.0%} train, {1 - train_ratio:.0%} eval")
        train_tasks, eval_tasks = split_tasks(all_tasks, train_ratio, seed)

    # Verify splits don't have duplicates
    if eval_tasks:
        train_ids = set(train_tasks.keys())
        eval_ids = set(eval_tasks.keys())
        overlap = train_ids & eval_ids
        if overlap:
            print(f"❌ ERROR: {len(overlap)} task IDs appear in both train and eval!")
            return 1
        print("✓ No overlap between train and eval sets")

    # Print statistics
    print_statistics(train_tasks, "Training Set")
    if eval_tasks:
        print_statistics(eval_tasks, "Evaluation Set")

    # Create ARC-format files
    print(f"\n{'=' * 60}")
    print("💾 Creating ARC-format files...")
    print(f"{'=' * 60}")

    # Training files
    print("Creating training files...")
    train_challenges, train_solutions = create_arc_format_files(
        train_tasks, output_dir, "concept_training"
    )
    print(f"✓ Created: {train_challenges.name}")
    print(f"✓ Created: {train_solutions.name}")

    # Evaluation files (if applicable)
    if eval_tasks:
        print("\nCreating evaluation files...")
        eval_challenges, eval_solutions = create_arc_format_files(
            eval_tasks, output_dir, "concept_evaluation"
        )
        print(f"✓ Created: {eval_challenges.name}")
        print(f"✓ Created: {eval_solutions.name}")

    print(f"\n{'=' * 60}")
    print("✅ Conversion complete!")
    print(f"{'=' * 60}")
    print(f"\nOutput files in: {output_dir.absolute()}")

    return 0


if __name__ == "__main__":
    main()
