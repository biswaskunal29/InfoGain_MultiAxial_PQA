import numpy as np
from pathlib import Path

FOLDER = r'E:\CVPR 23\Datasets\Our Dataset 5 Class\temp'


directory = '749003_2'
new_dir = Path(FOLDER,directory)
new_dir.mkdir(parents=True, exist_ok=True)


part1 = '_ocr'
part2 = '_desc'
part3 = '_labels'

filename = '749003_2' + part3 + '.npy'
filepath = new_dir / filename

np.save(filepath, np.array([[1, 2, 3],
                            [4, 5, 6],
                            [8, 9, 0]]))

a = np.load(filepath)

print(a)
























