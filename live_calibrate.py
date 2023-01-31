import numpy as np

# Load the array from a .npy file
intrinsic_params = np.load("intrinsic_params.npy")
distortion_params = np.load("distortion_params.npy")
rotational_params = np.load("rotational_params.npy")
translational_params = np.load("translational_params.npy")

print(intrinsic_params)
print(distortion_params)

#HELLOAOFOISRJFIOWJFEWFEFD