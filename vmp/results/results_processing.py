import json

import pandas as pd


def unzip_json_gz_folder(folder_path, remove_gz=False):
    """
    Unzip all .json.gz files in a folder to .json files.

    Args:
        folder_path (str or Path): Path to the folder containing .json.gz files
        remove_gz (bool): If True, remove the original .gz files after unzipping. Default is False.

    Returns:
        list: List of unzipped file paths
    """
    import gzip
    import shutil
    from pathlib import Path

    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    # Find all .json.gz files
    gz_files = list(folder.glob("*.json.gz"))

    if not gz_files:
        print(f"No .json.gz files found in {folder}")
        return []

    unzipped_files = []

    for gz_file in gz_files:
        # Create output filename by removing .gz extension
        output_file = gz_file.with_suffix("")

        # Skip if already unzipped
        if output_file.exists():
            print(f"Skipping {gz_file.name} - already unzipped")
            continue

        # Unzip the file
        print(f"Unzipping {gz_file.name}...")
        with gzip.open(gz_file, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        unzipped_files.append(output_file)

        # Optionally remove the original .gz file
        if remove_gz:
            gz_file.unlink()
            print(f"Removed {gz_file.name}")

    print(f"\nSuccessfully unzipped {len(unzipped_files)} files")
    return unzipped_files


def parse_trace_events_to_df(json_file_path, flatten_args=True):
    """
    Parse traceEvents from a PyTorch profiler JSON file into a pandas DataFrame.

    Args:
        json_file_path (str or Path): Path to the .json trace file
        flatten_args (bool): If True, flatten the 'args' dictionary into separate columns.
                            Default is True.

    Returns:
        pd.DataFrame: DataFrame containing the trace events
    """
    from pathlib import Path

    json_path = Path(json_file_path)

    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    # Load JSON file
    print(f"Loading {json_path.name}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract traceEvents
    trace_events = data.get("traceEvents", [])
    print(f"Found {len(trace_events)} trace events")

    if not trace_events:
        print("Warning: No trace events found in the file")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(trace_events)
    df["fname"] = json_path.name

    # Flatten args dictionary if requested
    if flatten_args and "args" in df.columns:
        # Extract args into separate columns
        args_df = pd.json_normalize(df["args"])
        args_df.columns = ["args_" + col for col in args_df.columns]

        # Drop original args column and concatenate flattened columns
        df = df.drop("args", axis=1)
        df = pd.concat([df, args_df], axis=1)

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df
