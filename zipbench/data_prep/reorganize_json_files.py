#!/usr/bin/env python3
"""
Reorganize JSON files.
Move every JSON file under the generated directory into a folder named after the file's original stem, and rename it to the given dataset name (default: simpleqa.json).
"""

import os
import sys
from pathlib import Path
import shutil
import argparse


def reorganize_json_files(target_dir: Path, dataset_name: str = "simpleqa", dry_run: bool = False):
    """
    Reorganize the JSON file layout.
    
    Args:
        target_dir: target directory path
        dataset_name: new file name (without extension)
        dry_run: if True, only print the planned operations without executing them
    """
    if not target_dir.exists():
        print(f"Error: directory does not exist: {target_dir}")
        return False
    
    if not target_dir.is_dir():
        print(f"Error: path is not a directory: {target_dir}")
        return False
    
    # Collect all JSON files
    json_files = list(target_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in directory {target_dir}")
        return False
    
    print(f"Found {len(json_files)} JSON file(s)")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    target_filename = f"{dataset_name}.json"
    
    for json_file in json_files:
        # File name without extension
        file_stem = json_file.stem
        
        # New folder path
        new_folder = target_dir / file_stem
        
        # New file path
        new_file_path = new_folder / target_filename
        
        print(f"\nProcessing: {json_file.name}")
        print(f"  -> create folder: {file_stem}/")
        print(f"  -> move and rename to: {file_stem}/{target_filename}")
        
        if dry_run:
            print("  [DRY RUN - not executed]")
            success_count += 1
            continue
        
        try:
            # Create the folder (skip if it already exists)
            new_folder.mkdir(exist_ok=True)
            
            # Check whether the target file already exists
            if new_file_path.exists():
                print(f"  Warning: target file already exists: {new_file_path}")
                print(f"  Skipping this file")
                error_count += 1
                continue
            
            # Move and rename the file
            shutil.move(str(json_file), str(new_file_path))
            print(f"  [OK] done")
            success_count += 1
            
        except Exception as e:
            print(f"  [FAIL] error: {e}")
            error_count += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Done.")
    print(f"Succeeded: {success_count} file(s)")
    if error_count > 0:
        print(f"Failed: {error_count} file(s)")
    
    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize JSON files: move each JSON file into a folder named after it and rename it to the given dataset name"
    )
    parser.add_argument(
        "target_dir",
        help="Target directory (the generated/ directory of the synthetic outputs)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="simpleqa",
        help="New file name (without extension); default: simpleqa"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned operations without executing them"
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the confirmation prompt and run directly"
    )
    
    args = parser.parse_args()
    
    target_dir = Path(args.target_dir)
    
    print("=" * 60)
    print("JSON file reorganization script")
    print("=" * 60)
    print(f"Target directory: {target_dir}")
    print(f"Target file name: {args.dataset}.json")
    print(f"Dry run mode: {'yes' if args.dry_run else 'no'}")
    print("=" * 60)
    
    # Confirmation prompt
    if not args.dry_run and not args.no_confirm:
        print("\nWarning: this operation will move and rename files!")
        response = input("Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Operation cancelled")
            return 0
    
    # Run the reorganization
    success = reorganize_json_files(target_dir, dataset_name=args.dataset, dry_run=args.dry_run)
    
    if args.dry_run:
        print("\nNote: this was a dry run. Re-run without --dry-run to apply the changes.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

