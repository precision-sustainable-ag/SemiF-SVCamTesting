import numpy as np
import cv2
import random
import shutil
from pathlib import Path

batch  = "NC_2024-11-27"
longeterm_storage = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-upload/", batch)

save_dir = Path("data/temp", batch)
save_dir.mkdir(exist_ok=True, parents=True)

raw_files = [x for x in longeterm_storage.glob("*.RAW")]
raw_files = sorted(list(longeterm_storage.glob("*.RAW")))

n = 40


last_n_files = raw_files[-n:]

for f in last_n_files:
    save_path = save_dir / f.name
    if not save_path.exists():
        shutil.copy(f, save_dir / f.name)