import argparse
import json
import sys
from pathlib import Path


def resolve_task_path(task_input: str) -> Path:
    direct_path = Path(task_input)
    if direct_path.exists():
        return direct_path

    filename = task_input if task_input.endswith(".json") else f"{task_input}.json"
    subpath = Path(filename)

    candidate_paths = [
        direct_path,
        Path("arc-agi") / subpath,
        Path("arc-agi") / "data" / subpath,
        Path("arc-agi") / "data" / "training" / subpath.name,
        Path("arc-agi") / "data" / "evaluation" / subpath.name,
        Path("arc-agi") / "data" / subpath.name,
        Path("data") / subpath,
        Path("data") / "training" / subpath.name,
        Path("data") / "evaluation" / subpath.name,
        Path("data") / subpath.name,
        Path(subpath.name),
    ]

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            return candidate

    return direct_path


def print_grid(grid: list[list[int]], title: str = ""):
    if title:
        print(f"  {title}:")
    for row in grid:
        print("    " + " ".join(f"{val}" for val in row))


def validate_task(task_path: Path, answer_data: dict, verbose: bool = True) -> tuple[bool, str]:
    with open(task_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    rule_summary = answer_data.get("rule_summary", "No rule summary provided")
    test_examples = task_data.get("test", [])
    predicted_outputs = answer_data.get("test_outputs", [])

    if verbose:
        print("========================================")
        print(f"Validating Task: {task_path.name}")
        print("========================================")
        print(f"Model Rule Summary:\n  \"{rule_summary}\"\n")

    if not isinstance(predicted_outputs, list):
        if verbose:
            print("Validation Failed: 'test_outputs' is missing or not a list.")
        return False, rule_summary

    if len(predicted_outputs) != len(test_examples):
        if verbose:
            print(
                f"Validation Failed: Expected {len(test_examples)} test outputs, "
                f"but found {len(predicted_outputs)}."
            )
        return False, rule_summary

    all_tests_passed = True

    for idx, (expected_example, pred_grid) in enumerate(zip(test_examples, predicted_outputs)):
        expected_grid = expected_example.get("output")
        if verbose:
            print(f"--- Test Case {idx + 1} ---")

        if expected_grid is None:
            if verbose:
                print("  (Warning: No ground truth 'output' in test data to validate against.)")
            continue

        if not isinstance(pred_grid, list) or not all(isinstance(r, list) for r in pred_grid):
            if verbose:
                print("  FAIL: Predicted grid is not a valid 2D array.")
            all_tests_passed = False
            continue

        match = pred_grid == expected_grid

        if match:
            if verbose:
                exp_shape = (len(expected_grid), len(expected_grid[0]) if expected_grid else 0)
                print(f"  Result: PASSED (Shape: {exp_shape[0]}x{exp_shape[1]})")
        else:
            all_tests_passed = False
            if verbose:
                exp_shape = (len(expected_grid), len(expected_grid[0]) if expected_grid else 0)
                pred_shape = (len(pred_grid), len(pred_grid[0]) if pred_grid else 0)
                print("  Result: FAILED")
                print(f"  Expected Shape: {exp_shape[0]}x{exp_shape[1]} | Predicted Shape: {pred_shape[0]}x{pred_shape[1]}")
                print_grid(expected_grid, "Expected Output")
                print_grid(pred_grid, "Predicted Output")

    if verbose:
        print("----------------------------------------")
        if all_tests_passed:
            print("OVERALL RESULT: SUCCESS")
        else:
            print("OVERALL RESULT: FAILED")
        print("========================================")

    return all_tests_passed, rule_summary


def main():
    parser = argparse.ArgumentParser(description="Validate ARC-AGI model solutions")
    parser.add_argument("task", help="Path or name of the task JSON file (e.g. 007bbfb7, data/training/007bbfb7.json)")
    parser.add_argument("answer", help="Path to the generated answer.json file (e.g. runs/example/answer.json)")

    args = parser.parse_args()

    task_path = resolve_task_path(args.task)
    if not task_path.exists():
        print(f"Error: Task file not found at '{args.task}'")
        sys.exit(1)

    answer_path = Path(args.answer)
    if not answer_path.exists():
        print(f"Error: Answer file not found at '{args.answer}'")
        sys.exit(1)

    with open(answer_path, "r", encoding="utf-8") as f:
        answer_data = json.load(f)

    is_valid, _ = validate_task(task_path, answer_data, verbose=True)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
