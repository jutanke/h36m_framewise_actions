import sys

sys.path.insert(0, "./../")

from h36m_fa.labels8 import get as get8
from h36m_fa.labels11 import get as get11
import h36m_fa.poses as poses


Lab8 = get8("S1", "walking", "1")
Lab11 = get11("S1", "walking", "1")

print("Lab8", Lab8.shape)
print("Lab11", Lab11.shape)


X = poses.get3d_fixed("S1", "walking", 1, data_dir="/home/tanke/Tmp/h36m")

print(X.shape)