import numpy as np
rows = np.indices((16, 4))[0]
print(rows)
cols = np.stack([np.arange(4) for _ in range(16)])
print(cols)
