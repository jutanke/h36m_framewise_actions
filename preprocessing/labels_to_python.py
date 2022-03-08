"""
converts the data labels into python code so that it can be easily used in
a python library (like this one!)

You DO NOT NEED TO CALL THIS to use this library or the data!

call this as follows:
```
(/{your_path}/h36m_framewise_actions)$ python preprocessing/labels_to_python.py
```
"""
import os
from os.path import isfile, isdir, join
import numpy as np
from tqdm import tqdm
from copy import deepcopy

r = input(
    "\nauto-generate the label python files\n-ONLY DO THIS IF YOU ARE THE MAINTAINER OF THIS LIBRARY-\ncontinue?[y/N]"
)
r = str(r)
if r != "y":
    print("good choice :)")
    print(r)
    exit()
print("generate data...")

cwd = os.getcwd()

actors = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
actions = [
    "directions",
    "discussion",
    "eating",
    "greeting",
    "phoning",
    "posing",
    "purchases",
    "sitting",
    "sittingdown",
    "smoking",
    "takingphoto",
    "waiting",
    "walking",
    "walkingdog",
    "walkingtogether",
]

PYTHON_FILE = [
    '"""',
    "THIS FILE HAS BEEN AUTO-GENERATED: DO NOT MODIFY!",
    '"""',
    "import numpy as np\n",
    "def get(actor, action, sid):",
    "\tglobal Labels",
    '\treturn Labels[f"{actor}_{action}_{sid}"]',
    "",
    "Labels = {",
]


PYTHON_FILE_8 = deepcopy(PYTHON_FILE)
PYTHON_FILE_11 = deepcopy(PYTHON_FILE)

for actor in tqdm(actors):
    for action in actions:
        for sid in [1, 2]:

            fname_8 = join(cwd, f"data/label8_{actor}_{action}_{sid}.txt")
            fname_11 = join(cwd, f"data/label11_{actor}_{action}_{sid}.txt")
            lab8 = np.loadtxt(fname_8).astype("int64").tolist()
            lab11 = np.loadtxt(fname_11).astype("int64").tolist()

            line8 = f'\t"{actor}_{action}_{sid}":np.array([{lab8}], dtype=np.int64),'
            line11 = f'\t"{actor}_{action}_{sid}":np.array([{lab11}], dtype=np.int64),'

            PYTHON_FILE_8.append(line8)
            PYTHON_FILE_11.append(line11)

PYTHON_FILE_8.append("}")
PYTHON_FILE_11.append("}")


PYTHON_FILE_8_fname = join(cwd, "h36m_fa/labels8.py")
PYTHON_FILE_11_fname = join(cwd, "h36m_fa/labels11.py")

with open(PYTHON_FILE_8_fname, "w") as f:
    for line in PYTHON_FILE_8:
        f.write(f"{line}\n")

with open(PYTHON_FILE_11_fname, "w") as f:
    for line in PYTHON_FILE_11:
        f.write(f"{line}\n")
