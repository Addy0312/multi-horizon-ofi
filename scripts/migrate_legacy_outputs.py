import os
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

for folder_name in ['results', 'model_weights']:
    folder_path = PROJECT_ROOT / folder_name
    if not folder_path.exists():
        continue

    # Get all files directly inside the folder (ignore directories)
    files = [f for f in folder_path.iterdir() if f.is_file()]

    for file_path in files:
        # Ignore dot files or metadata
        if file_path.name.startswith('.') or file_path.name == 'metadata.json':
            continue

        # Get modified time
        mtime = file_path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime)
        
        # Group by YYYY_MM_DD
        date_str = dt.strftime("%Y_%m_%d")
        
        dest_dir = folder_path / f"legacy_run_{date_str}"
        dest_dir.mkdir(exist_ok=True)
        
        print(f"Moving {file_path.name} to {dest_dir.name}/")
        shutil.move(str(file_path), str(dest_dir / file_path.name))

print("Legacy migration complete!")
