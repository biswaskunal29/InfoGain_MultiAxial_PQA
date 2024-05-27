
from pathlib import Path



#pathlib.Path(r"E:\CVPR 23\Datasets\Our Dataset 5 Class\temp").mkdir(parents=True, exist_ok=True)

FOLDER = r'E:\CVPR 23\Datasets\Our Dataset 5 Class\temp'

directory = 'foldy'

new_dir = Path(FOLDER,directory)
new_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# # You have to make a file inside the new directory
# new_file = new_dir / 'myfile2.txt'
# new_file.write_text('Hello file')
# 
# =============================================================================














