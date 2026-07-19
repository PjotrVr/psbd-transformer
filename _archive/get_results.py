import json
from pathlib import Path


def get_metrics_file_paths(root_directory):
    """
    Recursively finds all 'metrics.json' files within the target directory.
    """
    return Path(root_directory).rglob("metrics.json")


def load_json_content(file_path):
    """
    Handles the I/O operation of reading and parsing the JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (IOError, json.JSONDecodeError):
        # Why: Prevents the script from crashing if a file is locked or corrupted.
        return None


def extract_primary_stats(json_data):
    """
    Extracts the ASR and Clean Accuracy from the first entry of the metrics list.
    """
    if not isinstance(json_data, list) or len(json_data) == 0:
        return None

    first_entry = json_data[0]
    return {
        "asr": first_entry.get("asr_before"),
        "clean": first_entry.get("clean_accuracy"),
    }


def run_metrics_extraction(search_path):
    """
    Coordinates the file discovery and data extraction logic.
    """
    extracted_tuples = []

    for metrics_file in get_metrics_file_paths(search_path):
        data = load_json_content(metrics_file)
        stats = extract_primary_stats(data)

        if stats:
            # Why: .resolve().parent ensures we get the full absolute folder path.
            folder_path = str(metrics_file.resolve().parent)
            extracted_tuples.append((folder_path, stats["asr"], stats["clean"]))

    return extracted_tuples


def main():
    target_folder = "experiments/existing"

    # Linear orchestration of the logic
    results = run_metrics_extraction(target_folder)

    for entry in results:
        folder_path, asr, clean_acc = entry
        if folder_path.endswith("0_01"):
            print(f"({folder_path}, {asr:.3f}, {clean_acc:.3f})")


if __name__ == "__main__":
    main()
