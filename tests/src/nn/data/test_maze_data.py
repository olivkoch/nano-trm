#!/usr/bin/env python3
"""
Independent test script to validate pre-generated Maze datasets.

Reads a dataset from disk, prints samples, and validates correctness.
No dependencies on the main codebase - just numpy and click.

Validates:
- Start (S) and Goal (G) cells exist
- Solution path (o) connects S to G
- Path only traverses valid cells (not walls)

Usage:
    python test_maze_data.py ./data/maze-30x30-hard-1k
    python test_maze_data.py ./data/maze-30x30-hard-1k --num-samples 10
    python test_maze_data.py ./data/maze-30x30-hard-1k --validate-all
"""

import json
from pathlib import Path
from collections import deque

import click
import numpy as np


# Charset: "# SGo" -> 0=PAD, 1=#(wall), 2= (space), 3=S(start), 4=G(goal), 5=o(path)
CHARSET = "# SGo"
ID_TO_CHAR = {0: "?", 1: "#", 2: " ", 3: "S", 4: "G", 5: "o"}
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARSET)}

# Token IDs
WALL = 1
SPACE = 2
START = 3
GOAL = 4
PATH = 5


def load_dataset(data_dir: Path, split: str = "train"):
    """Load a dataset split from disk."""
    split_dir = data_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy")
    
    with open(split_dir / "dataset.json") as f:
        metadata = json.load(f)
    
    return inputs, labels, metadata


def load_global_metadata(data_dir: Path):
    """Load global metadata from the dataset."""
    metadata_path = data_dir / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path) as f:
        return json.load(f)


def decode_grid(encoded: np.ndarray, grid_size: int) -> np.ndarray:
    """Decode an encoded sequence back to a 2D grid."""
    return encoded.reshape(grid_size, grid_size)


def grid_to_str(grid: np.ndarray) -> str:
    """Convert grid of token IDs to string representation."""
    result = []
    for row in grid:
        result.append("".join(ID_TO_CHAR.get(v, "?") for v in row))
    return "\n".join(result)


def find_cell(grid: np.ndarray, token_id: int):
    """Find coordinates of a cell with given token ID. Returns (row, col) or None."""
    positions = np.argwhere(grid == token_id)
    if len(positions) == 0:
        return None
    return tuple(positions[0])


def find_all_cells(grid: np.ndarray, token_id: int):
    """Find all coordinates of cells with given token ID."""
    return [tuple(pos) for pos in np.argwhere(grid == token_id)]


def is_valid_maze(grid: np.ndarray):
    """
    Check if a maze grid is structurally valid.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for exactly one start
    starts = find_all_cells(grid, START)
    if len(starts) == 0:
        return False, "No start cell (S) found"
    if len(starts) > 1:
        return False, f"Multiple start cells found: {starts}"
    
    # Check for exactly one goal
    goals = find_all_cells(grid, GOAL)
    if len(goals) == 0:
        return False, "No goal cell (G) found"
    if len(goals) > 1:
        return False, f"Multiple goal cells found: {goals}"
    
    return True, "Valid structure"


def validate_solution_path(puzzle: np.ndarray, solution: np.ndarray):
    """
    Validate that the solution contains a valid path from S to G.
    
    Returns:
        Tuple of (is_valid, error_message, path_length)
    """
    grid_size = puzzle.shape[0]
    
    # Find start and goal
    start = find_cell(solution, START)
    goal = find_cell(solution, GOAL)
    
    if start is None:
        return False, "Solution missing start cell (S)", 0
    if goal is None:
        return False, "Solution missing goal cell (G)", 0
    
    # Check that path cells don't overwrite walls
    path_cells = find_all_cells(solution, PATH)
    for r, c in path_cells:
        if puzzle[r, c] == WALL:
            return False, f"Path goes through wall at ({r}, {c})", 0
    
    # BFS to verify path connectivity from S to G through path cells
    # Valid moves: cells that are PATH, START, or GOAL
    valid_cells = {START, GOAL, PATH, SPACE}
    
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        r, c = queue.popleft()
        
        if (r, c) == goal:
            return True, "Valid path", len(path_cells)
        
        # Check 4-connected neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) not in visited:
                    cell_val = solution[nr, nc]
                    # Can traverse PATH cells, or reach GOAL
                    if cell_val == PATH or cell_val == GOAL:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
    
    return False, "Path does not connect S to G", len(path_cells)


def solution_preserves_puzzle(puzzle: np.ndarray, solution: np.ndarray):
    """Check that solution preserves walls and key cells from puzzle."""
    # Walls should remain walls
    puzzle_walls = puzzle == WALL
    solution_walls = solution == WALL
    
    if not np.all(puzzle_walls == solution_walls):
        return False, "Solution modified wall structure"
    
    # Start and goal should be in same position
    puzzle_start = find_cell(puzzle, START)
    solution_start = find_cell(solution, START)
    if puzzle_start != solution_start:
        return False, f"Start moved from {puzzle_start} to {solution_start}"
    
    puzzle_goal = find_cell(puzzle, GOAL)
    solution_goal = find_cell(solution, GOAL)
    if puzzle_goal != solution_goal:
        return False, f"Goal moved from {puzzle_goal} to {solution_goal}"
    
    return True, "Preserved"


def print_grid(grid: np.ndarray, title: str = "", colorize: bool = True):
    """Pretty print a maze grid with optional ANSI colors."""
    if title:
        print(title)
    
    grid_size = grid.shape[0]
    
    # Color codes
    COLORS = {
        WALL: "\033[90m",    # Dark gray for walls
        SPACE: "\033[0m",    # Default for space
        START: "\033[92m",   # Green for start
        GOAL: "\033[91m",    # Red for goal
        PATH: "\033[93m",    # Yellow for path
    }
    RESET = "\033[0m"
    
    # Top border
    print("+" + "-" * (grid_size * 2) + "+")
    
    for row in grid:
        line = "|"
        for val in row:
            char = ID_TO_CHAR.get(val, "?")
            if colorize:
                color = COLORS.get(val, "")
                line += f"{color}{char}{RESET} "
            else:
                line += f"{char} "
        line = line.rstrip() + "|"
        print(line)
    
    # Bottom border
    print("+" + "-" * (grid_size * 2) + "+")


def print_side_by_side(puzzle: np.ndarray, solution: np.ndarray, grid_size: int, max_width: int = 40):
    """Print puzzle and solution side by side (for smaller mazes)."""
    if grid_size > max_width // 2:
        # Too wide, print vertically
        print_grid(puzzle, "\nPuzzle:")
        print_grid(solution, "\nSolution:")
        return
    
    print("\nPuzzle:" + " " * (grid_size * 2 - 3) + "  Solution:")
    
    # Top border
    border = "+" + "-" * (grid_size * 2) + "+"
    print(f"{border}  {border}")
    
    COLORS = {
        WALL: "\033[90m", SPACE: "\033[0m", START: "\033[92m",
        GOAL: "\033[91m", PATH: "\033[93m",
    }
    RESET = "\033[0m"
    
    for p_row, s_row in zip(puzzle, solution):
        p_line = "|"
        s_line = "|"
        for val in p_row:
            char = ID_TO_CHAR.get(val, "?")
            p_line += f"{COLORS.get(val, '')}{char}{RESET} "
        for val in s_row:
            char = ID_TO_CHAR.get(val, "?")
            s_line += f"{COLORS.get(val, '')}{char}{RESET} "
        print(f"{p_line.rstrip()}|  {s_line.rstrip()}|")
    
    print(f"{border}  {border}")


def print_sample(idx: int, puzzle: np.ndarray, solution: np.ndarray, grid_size: int):
    """Print a single sample with puzzle and solution."""
    print(f"\n{'='*60}")
    print(f"Sample {idx} (grid_size={grid_size}x{grid_size})")
    print(f"{'='*60}")
    
    # Stats
    num_walls = np.sum(puzzle == WALL)
    num_spaces = np.sum(puzzle == SPACE)
    num_path = np.sum(solution == PATH)
    
    start = find_cell(puzzle, START)
    goal = find_cell(puzzle, GOAL)
    
    print(f"Walls: {num_walls}, Open spaces: {num_spaces}, Path length: {num_path}")
    print(f"Start: {start}, Goal: {goal}")
    
    print_side_by_side(puzzle, solution, grid_size)


def validate_sample(puzzle: np.ndarray, solution: np.ndarray):
    """Validate a single sample. Returns list of errors (empty if valid)."""
    errors = []
    
    # Check puzzle structure
    valid, msg = is_valid_maze(puzzle)
    if not valid:
        errors.append(f"Invalid puzzle: {msg}")
    
    # Check solution structure
    valid, msg = is_valid_maze(solution)
    if not valid:
        errors.append(f"Invalid solution: {msg}")
    
    # Check solution preserves puzzle
    valid, msg = solution_preserves_puzzle(puzzle, solution)
    if not valid:
        errors.append(f"Preservation error: {msg}")
    
    # Check path validity
    valid, msg, path_len = validate_solution_path(puzzle, solution)
    if not valid:
        errors.append(f"Path error: {msg}")
    
    return errors


@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="train",
    help="Which split to test (default: train)",
)
@click.option(
    "--num-samples",
    type=int,
    default=5,
    help="Number of samples to print (default: 5)",
)
@click.option(
    "--validate-all",
    is_flag=True,
    default=False,
    help="Validate all samples (not just printed ones)",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Only print summary, not individual samples",
)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="Disable ANSI color output",
)
def main(data_dir: Path, split: str, num_samples: int, validate_all: bool, quiet: bool, no_color: bool):
    """
    Test and validate pre-generated Maze datasets.
    
    DATA_DIR is the path to the pre-generated dataset directory.
    """
    # Load global metadata
    click.echo(f"Loading dataset from: {data_dir}")
    metadata = load_global_metadata(data_dir)
    
    click.echo(f"\nGlobal metadata:")
    click.echo(f"  Mode: {metadata.get('mode', 'maze')}")
    click.echo(f"  Grid size: {metadata['max_grid_size']}x{metadata['max_grid_size']}")
    click.echo(f"  Vocab size: {metadata['vocab_size']}")
    click.echo(f"  Sequence length: {metadata['seq_len']}")
    click.echo(f"  Train/Val/Test: {metadata['num_train']}/{metadata['num_val']}/{metadata['num_test']}")
    
    # Load split
    click.echo(f"\nLoading {split} split...")
    inputs, labels, split_meta = load_dataset(data_dir, split)
    
    total_samples = len(inputs)
    grid_size = split_meta["grid_size"]
    
    click.echo(f"\nSplit metadata:")
    click.echo(f"  Samples: {total_samples}")
    click.echo(f"  Grid size: {grid_size}x{grid_size}")
    
    # Print some samples
    if not quiet:
        print_indices = np.linspace(0, total_samples - 1, min(num_samples, total_samples), dtype=int)
        
        for idx in print_indices:
            puzzle = decode_grid(inputs[idx], grid_size)
            solution = decode_grid(labels[idx], grid_size)
            print_sample(idx, puzzle, solution, grid_size)
            
            errors = validate_sample(puzzle, solution)
            if errors:
                click.echo("❌ ERRORS:")
                for err in errors:
                    click.echo(f"   - {err}")
            else:
                click.echo("✓ Valid")
    
    # Validate all samples if requested
    if validate_all:
        click.echo(f"\n{'='*60}")
        click.echo(f"Validating ALL {total_samples} samples...")
        click.echo(f"{'='*60}")
        
        invalid_count = 0
        error_details = []
        path_lengths = []
        
        for idx in range(total_samples):
            puzzle = decode_grid(inputs[idx], grid_size)
            solution = decode_grid(labels[idx], grid_size)
            
            errors = validate_sample(puzzle, solution)
            if errors:
                invalid_count += 1
                if len(error_details) < 10:  # Keep first 10 errors
                    error_details.append((idx, errors))
            else:
                # Track path length for valid samples
                path_lengths.append(np.sum(solution == PATH))
            
            if (idx + 1) % 1000 == 0:
                click.echo(f"  Validated {idx + 1}/{total_samples}...")
        
        click.echo(f"\n{'='*60}")
        click.echo(f"VALIDATION SUMMARY")
        click.echo(f"{'='*60}")
        click.echo(f"Total samples: {total_samples}")
        click.echo(f"Valid samples: {total_samples - invalid_count}")
        click.echo(f"Invalid samples: {invalid_count}")
        
        if path_lengths:
            click.echo(f"\nPath length statistics (valid samples):")
            click.echo(f"  Min: {min(path_lengths)}")
            click.echo(f"  Max: {max(path_lengths)}")
            click.echo(f"  Mean: {np.mean(path_lengths):.1f}")
            click.echo(f"  Median: {np.median(path_lengths):.1f}")
        
        if invalid_count == 0:
            click.echo("\n✅ ALL SAMPLES VALID!")
        else:
            click.echo(f"\n❌ {invalid_count} INVALID SAMPLES")
            click.echo("\nFirst few errors:")
            for idx, errors in error_details:
                click.echo(f"\n  Sample {idx}:")
                for err in errors:
                    click.echo(f"    - {err}")
    
    # Check for duplicates
    click.echo(f"\n{'='*60}")
    click.echo("Checking for duplicates...")
    click.echo(f"{'='*60}")
    
    puzzle_hashes = set()
    duplicate_count = 0
    
    for idx in range(total_samples):
        h = inputs[idx].tobytes()
        if h in puzzle_hashes:
            duplicate_count += 1
        else:
            puzzle_hashes.add(h)
    
    if duplicate_count == 0:
        click.echo(f"✓ No duplicates found in {split} split")
    else:
        click.echo(f"⚠ Found {duplicate_count} duplicate puzzles in {split} split")
    
    click.echo("\n✅ Test complete!")


if __name__ == "__main__":
    main()