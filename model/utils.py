from pathlib import Path
import shutil

def make_checkpoint_dir(dir_checkpoint):
        
    path = Path(dir_checkpoint)
    # create folder if it does not exist
    if not path.exists():
        # shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=False)