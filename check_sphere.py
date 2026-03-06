import numpy as np
data = np.load("/factored_neus/public_data/sleeve4_1mf/cameras_sphere.npz")

for i in range(5):
    P = data[f"world_mat_{i}"]
    print(P)