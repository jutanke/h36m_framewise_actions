"""
Call this function as follows:
```
```
(/{your_path}/h36m_framewise_actions)$ python preprocessing/process_h36m.py {/path/to/orginal/h36m_dir} {/target/data/dir}
```
```
"""
from spacepy import pycdf
import numpy as np
from tqdm import tqdm


import sys
from os.path import isfile, isdir, join
from os import listdir, makedirs
import requests
import zipfile

assert len(sys.argv) == 3

path_source = sys.argv[1]
path_destination = sys.argv[2]

assert isdir(path_source)
if not isdir(path_destination):
    makedirs(path_destination)

print(f"Process H36M from {path_source} to {path_destination}")

# --- generate data ---
ACTORS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
ACTIONS = [
    "Directions",
    "Discussion",
    "Eating",
    "Greeting",
    "Phoning",
    "Posing",
    "Purchases",
    "Sitting",
    "SittingDown",
    "Smoking",
    "Photo",
    "Waiting",
    "Walking",
    "WalkingDog",
    "WalkTogether",
]

# get EXPMAPS
zip_fname = join(path_destination, "h3.6m.zip")
if not isfile(zip_fname):
    print("[data aquisition] - h36m - download expmap data")
    r = requests.get("http://www.cs.stanford.edu/people/ashesh/h3.6m.zip")
    open(zip_fname, "wb").write(r.content)
exp_dir = join(path_destination, "exp_dir")
exp_data_dir = join(exp_dir, "h3.6m")
if not isdir(exp_data_dir):
    print("[data aquisition] - h36m - extract exmap data")
    with zipfile.ZipFile(zip_fname, "r") as zip_ref:
        zip_ref.extractall(exp_dir)

for actor in ACTORS:
    print("\n[get h36m skeleton] ->", actor)
    for action in tqdm(ACTIONS):
        for sid in [0, 1]:
            # fix labeling... Human3.6M labeling is very messy and we need to fix it...

            fixed_action = ""
            if action == "WalkTogether":
                fixed_action = "walkingtogether"
            elif action == "Photo":
                fixed_action = "takingphoto"
            else:
                fixed_action = action.lower()

            if actor == "S1" and action == "Photo":
                action = "TakingPhoto"
            if actor != "S1" and action == "WalkingDog":
                action = "WalkDog"

            cdf_dir = join(join(path_source, actor), "MyPoseFeatures")
            cdf_dir = join(cdf_dir, "D3_Positions")

            videos = sorted([f for f in listdir(cdf_dir) if f.startswith(action)])

            if (actor == "S1" and action == "Walking") or action == "Sitting":
                # separate Walking from WalkingDog OR
                # separate Sitting from SittingDown
                assert len(videos) == 4
                videos = videos[0:2]

            assert len(videos) == 2, "# of videos:" + str(len(videos))
            a, b = videos
            if len(a) > len(b):  # ['xxx 9.cdf', 'xxx.cdf']
                videos = [b, a]
            else:
                assert len(a) == len(b)

            cdf_file = join(cdf_dir, videos[sid])
            assert isfile(cdf_file)

            cdf = pycdf.CDF(cdf_file)
            joints3d = np.squeeze(cdf["Pose"]).reshape((-1, 32, 3)) / 1000
            joints3d = joints3d.astype("float32")

            # more magic to harmonize the naming conventions...
            fixed_sid = sid + 1
            fixed_sid = fixed_sid % 2 + 1  # 2 -> 1 and 1 -> 2

            if (
                (actor == "S8" and fixed_action == "walkingtogether")
                or (actor == "S7" and fixed_action == "walking")
                or (actor == "S7" and fixed_action == "waiting")
                or (actor == "S5" and fixed_action == "waiting")
                or (actor == "S7" and fixed_action == "takingphoto")
                or (actor == "S6" and fixed_action == "takingphoto")
                or (actor == "S5" and fixed_action == "takingphoto")
                or (actor == "S11" and fixed_action == "sittingdown")
                or (actor == "S9" and fixed_action == "sittingdown")
                or (actor == "S8" and fixed_action == "sittingdown")
                or (actor == "S7" and fixed_action == "sittingdown")
                or (actor == "S5" and fixed_action == "sittingdown")
                or (actor == "S6" and fixed_action == "sitting")
                or (actor == "S1" and fixed_action == "sitting")
                or (actor == "S5" and fixed_action == "greeting")
                or (actor == "S6" and fixed_action == "eating")
                or (actor == "S11" and fixed_action == "discussion")
                or (actor == "S9" and fixed_action == "discussion")
                or (actor == "S5" and fixed_action == "discussion")
                or (actor == "S5" and fixed_action == "directions")
            ):
                fixed_sid = fixed_sid % 2 + 1  # 2 -> 1 and 1 -> 2

            if fixed_action == "walkdog":
                fixed_action = "walkingdog"

            fname = join(
                path_destination,
                actor + "_" + fixed_action + "_" + str(fixed_sid) + ".npy",
            )
            np.save(fname, joints3d)
