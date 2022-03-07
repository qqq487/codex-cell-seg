import cv2
import numpy as np
import sys
from cellpose import dynamics, io, plot, utils
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure


mask = np.load("../train_data/multi_refined_same_ch_wo23/masks/13_mask.npy")

# labeled_array, num_features = label(mask)

mask = mask.astype('uint')

res = utils.distance_to_boundary(mask)

print("distance_to_boundary mask = ",res)