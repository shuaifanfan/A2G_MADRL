class Setting(object):
    def __init__(self, log):
        self.V = {
            'MAP_X': 16,
            'MAP_Y': 16,
            'MAX_VALUE': 1.,
            'MIN_VALUE': 0.,
            'DATA': [[9.9140119553e-01, 2.2311273217e-01, 5.9645938873e-01],
                     [3.4761524200e-01, 7.4505066872e-01, 7.2536742687e-01],
                     [7.5244140625e-01, 5.2246093750e-01, 3.8623046875e-01],
                     [5.3728169203e-01, 7.5541979074e-01, 3.0111008883e-01],
                     [2.2888183594e-01, 9.4384765625e-01, 8.8964843750e-01],
                     [6.2519145012e-01, 1.9104470313e-01, 6.8883723021e-01],
                     [8.9111328125e-01, 2.1118164062e-01, 9.6777343750e-01],
                     [8.4932959080e-01, 7.8816157579e-01, 5.8413469791e-01],
                     [1.2944786251e-01, 1.9150093198e-02, 1.0947148502e-01],
                     [3.3258455992e-01, 9.3344014883e-01, 2.4454586208e-01],
                     [4.5524990559e-01, 3.1826683879e-01, 6.6131256521e-02],
                     [9.5935279131e-01, 5.6751412153e-01, 8.2611012459e-01],
                     [3.3983018994e-01, 5.2995896339e-01, 3.7023225427e-01],
                     [1.5314593911e-01, 4.7007226944e-01, 5.7440057397e-02],
                     [6.1045295000e-01, 9.0187722445e-01, 9.3121069670e-01],
                     [3.8273200393e-01, 1.8045753241e-01, 9.0950518847e-01],
                     [3.0590510368e-01, 5.2689367533e-01, 3.2359883189e-01],
                     [7.4568825960e-01, 6.7214512825e-01, 8.2028371096e-01],
                     [4.7210693359e-02, 4.6972656250e-01, 4.1479492188e-01],
                     [6.0888671875e-01, 7.6806640625e-01, 7.3291015625e-01],
                     [3.3881655335e-01, 3.6587712821e-03, 8.1039929390e-01],
                     [6.8050384521e-02, 4.6217329800e-02, 8.6730951071e-01],
                     [6.4951008558e-01, 5.8001291752e-01, 9.5590710640e-01],
                     [2.2898811102e-01, 6.8988484144e-01, 9.4617402554e-01],
                     [6.6210937500e-01, 5.5273437500e-01, 7.2631835938e-02],
                     [1.4587204158e-01, 1.1123943329e-01, 5.8969140053e-01],
                     [9.9600201845e-01, 8.9279687405e-01, 2.3184943199e-01],
                     [9.0402406454e-01, 7.1088336408e-02, 5.6631070375e-01],
                     [4.2725309730e-01, 3.2270619273e-01, 7.0968106389e-02],
                     [1.8604320288e-01, 2.1669780836e-02, 1.1593785882e-01],
                     [2.8540039062e-01, 2.2644042969e-01, 8.5791015625e-01],
                     [6.6345214844e-02, 9.2041015625e-01, 8.7585449219e-02],
                     [5.4874485731e-01, 8.4172517061e-02, 5.5937552452e-01],
                     [1.3497097790e-01, 2.5999665260e-01, 4.8310092092e-01],
                     [6.3352006674e-01, 9.9910354614e-01, 2.9034343362e-01],
                     [2.8289541602e-01, 6.3986515999e-01, 8.8266563416e-01],
                     [8.3301407099e-01, 1.3167610765e-01, 3.7671697140e-01],
                     [5.6819146872e-01, 5.7002061605e-01, 9.6483540535e-01],
                     [2.1766318381e-01, 8.2596439123e-01, 3.6273404956e-01],
                     [3.3583003283e-01, 1.4563038945e-01, 9.2138433456e-01],
                     [2.0942579210e-01, 6.1259585619e-01, 9.1531115770e-01],
                     [9.5166015625e-01, 1.3305664062e-01, 8.0810546875e-01],
                     [6.7824435234e-01, 9.8050266504e-01, 9.9353903532e-01],
                     [8.6989158392e-01, 2.8915050626e-01, 2.9618918896e-01],
                     [2.4657636881e-01, 9.0613758564e-01, 7.9670095444e-01],
                     [3.1152343750e-01, 9.4921875000e-01, 3.2446289062e-01],
                     [5.8995997906e-01, 4.1139656305e-01, 1.7636056244e-01],
                     [7.4824392796e-02, 4.7780549526e-01, 4.8981487751e-01],
                     [5.7060468197e-01, 5.6363576651e-01, 4.3456125259e-01],
                     [1.2463040650e-01, 3.6579437554e-02, 7.6846516132e-01]],

            'OBSTACLE': [
                [0, 4, 1, 1],
                [0, 9, 1, 1],
                [0, 10, 2, 1],
                [2, 2, 2, 1],
                [5, 13, 1, 1],
                [6, 12, 2, 1],
                [10, 5, 3, 1],
                [11, 5, 1, 3],
                [10, 13, 1, 2],
                [11, 13, 2, 1],
                [12, 0, 1, 2],
                [12, 5, 1, 1],
                [12, 7, 1, 1],
                [15, 11, 1, 1]
            ],

            'STATION': [
                [0.75, 0.625],
                [0.25, 0.25],
                [0.0625, 0.9375],
                [0.9375, 0.0625],
                [0.875, 0.875]
            ],
            'CHANNEL': 3,

            'NUM_UAV': 2,
            'INIT_POSITION': (0, 10, 10),
            'MAX_ENERGY': 30.,  # must face the time of lack
            'NUM_ACTION': 2,  # 2
            'SAFE_ENERGY_RATE': 0.2,
            'RANGE': 1.1,
            'MAXDISTANCE': 1.,
            'COLLECTION_PROPORTION': 0.2,  # c speed
            'FILL_PROPORTION': 0.2,  # fill speed

            'WALL_REWARD': -1.,
            'VISIT': 1. / 1000.,
            'DATA_REWARD': 1.,
            'FILL_REWARD': 1.,
            'ALPHA': 1.,
            'BETA': 0.1,
            'EPSILON': 1e-4,
            'NORMALIZE': .1,
            'FACTOR': 0.1,
            'DiscreteToContinuous': [
                [-1.0, -1.0],
                [-1.0, -0.5],
                [-1.0, 0.0],
                [-1.0, 0.5],
                [-1.0, 1.0],
                [-0.5, -1.0],
                [-0.5, -0.5],
                [-0.5, 0.0],
                [-0.5, 0.5],
                [-0.5, 1.0],
                [0.0, -1.0],
                [0.0, -0.5],
                [0.0, 0.0],  # 重点关注no-op
                [0.0, 0.5],
                [0.0, 1.0],
                [0.5, -1.0],
                [0.5, -0.5],
                [0.5, 0.0],
                [0.5, 0.5],
                [0.5, 1.0],
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],

            ]
        }
        self.LOG = log
        if self.LOG is not None:
            self.time = log.time

    def log(self):
        if self.LOG is not None:
            self.LOG.log(self.V)
        else:
            pass
