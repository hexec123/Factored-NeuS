import numpy as np
import matplotlib.pyplot as plt

data = np.load("/factored_neus/public_data/sleeve4_1mf/cameras_sphere.npz")

cams = []

for k in data.files:
    if "world_mat" in k:
        P = data[k]
        cam = -np.linalg.inv(P[:3,:3]) @ P[:3,3]
        cams.append(cam)

cams = np.array(cams)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(cams[:,0], cams[:,1], cams[:,2])
plt.show()