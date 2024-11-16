from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from noma_utils import compute_capacity_G2A, compute_channel_gain_G2G

size = 2000


class TestCommModel(TestCase):
    noma_config = {
        'noise0_density': 5e-20,
        'bandwidth_subchannel': 20e6 / 5,
        'p_uav': 4,  # w, 也即34.7dbm
        'p_poi': 0.1,
        'aA': 2,
        'aG': 3,
        'nLoS': 0,  # dB, 也即1w
        'nNLoS': -20,  # dB, 也即0.01w
        'uav_init_height': 50,
        'psi': 9.6,
        'beta': 0.16,
    }

    def test_compute_capacity_g2a(self):
        x, y = np.meshgrid(np.arange(0, size, dtype='f'), np.arange(0, size, dtype='f'))
        z = np.arange(0, size * size, dtype='f')
        for index, (main_poi_dis, interfere) in enumerate(zip(x.flat, y.flat)):
            sinr_G2A, R_G2A = compute_capacity_G2A(self.noma_config, main_poi_dis, interfere)
            z[index] = R_G2A
        ax = plt.axes(projection="3d")
        ax.scatter3D(x.flat, y.flat, z, c=z, cmap='cividis')

    def test_poi_j_influence(self):
        # poi_j can cause more than 5x of interference at 5km, cannot be ignored.
        ratios = []
        n0 = self.noma_config['noise0_density']
        B0 = self.noma_config['bandwidth_subchannel']
        P_poi = self.noma_config['p_poi']  # w
        env_noise = n0 * B0
        my_range = range(2000, 10000, 5)
        for poi_j_dis in my_range:
            Gj_G2G = compute_channel_gain_G2G(self.noma_config, poi_j_dis)
            ratio = (env_noise + Gj_G2G * P_poi) / (env_noise)
            ratios.append(ratio)
        fig, ax = plt.subplots()
        ax.plot(my_range, ratios)
        ax.axhline(y=5, color='r', linestyle='--', label='Horizontal Line at y=4')
        plt.show()
