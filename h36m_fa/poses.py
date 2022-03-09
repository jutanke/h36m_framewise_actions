import numpy as np
from os.path import join, isdir, isfile, abspath
from os import makedirs
import h36m_fa.conversion as conv
import h36m_fa.fk as FK
import h36m_fa.kabsch as KB
from h36m_fa.mirror import reflect_over_x, mirror_p3d

from tqdm import tqdm


ACTORS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
ACTIONS = [
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


def get3d(actor: str, action: str, sid: int, data_dir: str):
    """
    Returns the official Human3.6M dataset 3D keypoints.
    All joints are in global coordinates and the skeletons
    are sized according to the actor.
    """
    data_dir = abspath(data_dir)
    fname = join(data_dir, f"{actor}_{action}_{sid}.npy")
    return np.load(fname)


def get3d_fixed(actor, action, sid, data_dir: str):
    """"""
    data_dir = abspath(data_dir)
    acquire_fixed_skeleton(data_dir)
    fname_binary = join(
        join(data_dir, "fixed_skeleton"), actor + "_" + action + "_" + str(sid) + ".npy"
    )
    if isfile(fname_binary):
        seq = np.load(fname_binary)
    else:
        fname = join(
            join(data_dir, "fixed_skeleton"),
            actor + "_" + action + "_" + str(sid) + ".txt",
        )
        seq = np.loadtxt(fname, dtype=np.float32)
        np.save(fname_binary, seq)
    return seq


def get_expmap(actor, action, sid, data_dir: str):
    """
    ExpMap representation as provided in Martinez et al.
    """
    data_dir = abspath(data_dir)
    fname = join(
        join(join(data_dir, "exp_dir/h3.6m/dataset"), actor),
        action + "_" + str(sid) + ".txt",
    )
    if not isfile(fname):
        print(f"cannot find {fname}")
        print("Did you forget to extract Human3.6M?")
        print("preprocessing/process_h36m.py")
        print("exiting...")
        exit()
    seq = np.loadtxt(fname, delimiter=",", dtype=np.float32)
    return seq


def get_euler(actor, action, sid, data_dir: str):
    """
    Euler Angle representation.
    """
    data_dir = abspath(data_dir)
    acquire_euler(data_dir)
    fname = join(
        join(data_dir, "euler"), actor + "_" + action + "_" + str(sid) + ".npy"
    )
    return np.load(fname)


CACHE_get3d_fixed_from_rotation = {}


def get3d_fixed_from_rotation(actor, action, sid, data_dir):
    loc = join(data_dir, "fixed_skeleton_from_rotation")
    fname = join(loc, actor + "_" + action + "_" + str(sid) + ".txt")
    if isfile(fname):
        seq = np.load(fname)
        n_frames = len(seq)
        seq = seq.reshape((n_frames, -1))
        return seq
    else:
        if not isdir(loc):
            makedirs(loc)
        global CACHE_get3d_fixed_from_rotation
        if (actor, action, sid) not in CACHE_get3d_fixed_from_rotation:
            seq = get_euler(actor, action, sid, data_dir)
            seq = FK.euler_fk(seq)
            seq = reflect_over_x(seq)
            seq = mirror_p3d(
                seq
            )  # there are some mirroring issues in the original rotational data:
            # https://github.com/una-dinosauria/human-motion-prediction/issues/46
            seq = seq.astype("float32")
            np.save(fname, seq)
            n_frames = len(seq)
            seq = seq.reshape((n_frames, -1))
            CACHE_get3d_fixed_from_rotation[actor, action, sid] = seq
        return CACHE_get3d_fixed_from_rotation[actor, action, sid]


# ==============================
# A C Q U I R E
# ==============================


def acquire_fixed_skeleton(data_dir: str):
    data_dir_orig = data_dir
    acquire_euler(data_dir)
    data_dir = join(data_dir, "fixed_skeleton")
    if not isdir(data_dir):
        print("[mocap][Human3.6M] generate fixed skeletons:", data_dir)
        makedirs(data_dir)
        for actor in ACTORS:
            print("\thandle actor ", actor)
            for action in tqdm(ACTIONS):
                for sid in [1, 2]:
                    fname = join(
                        data_dir, actor + "_" + action + "_" + str(sid) + ".txt"
                    )
                    seq1 = get3d_fixed_from_rotation(actor, action, sid, data_dir_orig)
                    seq2 = get3d(actor, action, sid, data_dir_orig)
                    assert len(seq1) == len(seq2), (
                        actor
                        + " "
                        + action
                        + " -> "
                        + str(seq1.shape)
                        + "|"
                        + str(seq2.shape)
                    )

                    seq1_ = []
                    for p, q in zip(seq1, seq2):
                        p = KB.rotate_P_to_Q(p, q)
                        seq1_.append(p)
                    n_frames = len(seq1)
                    seq1 = np.reshape(seq1_, (n_frames, -1))
                    np.savetxt(fname, seq1)


def acquire_fixed_skeleton_from_rotation(data_dir: str):
    acquire_euler(data_dir)
    data_dir = join(data_dir, "fixed_skeleton_from_rotation")
    if not isdir(data_dir):
        print("[mocap][Human3.6M] generate fixed skeletons from rotation:", data_dir)
        makedirs(data_dir)
        for actor in ACTORS:
            print("\thandle actor ", actor)
            for action in tqdm(ACTIONS):
                for sid in [1, 2]:
                    fname = join(
                        data_dir, actor + "_" + action + "_" + str(sid) + ".npy"
                    )
                    seq1 = get3d_fixed_from_rotation(actor, action, sid, data_dir)
                    np.save(fname, seq1)


def acquire_euler(data_dir: str):
    euler_dir = join(data_dir, "euler")
    if not isdir(euler_dir):
        makedirs(euler_dir)
    for actor in ACTORS:
        for action in ACTIONS:
            for sid in [1, 2]:
                fname = join(euler_dir, actor + "_" + action + "_" + str(sid) + ".npy")
                if not isfile(fname):
                    print(
                        "[data aquisition] - h36m - extract euler ",
                        (actor, action, sid),
                    )
                    exp_seq = get_expmap(actor, action, sid, data_dir)
                    euler_seq = conv.expmap2euler(exp_seq).astype("float32")
                    np.save(fname, euler_seq)