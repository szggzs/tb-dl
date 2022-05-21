import numpy as np


def mesh3D(box_len=5, num_grids=101):

    x = np.linspace(-box_len, box_len, num=num_grids)
    y = np.linspace(-box_len, box_len, num=num_grids)
    z = np.linspace(-box_len, box_len, num=num_grids)

    x = np.expand_dims(np.expand_dims(x, -1), -1)
    y = np.expand_dims(np.expand_dims(y, 0), -1)
    z = np.expand_dims(np.expand_dims(z, 0), 0)

    x = x.repeat(num_grids, axis=1).repeat(num_grids, axis=2)
    y = y.repeat(num_grids, axis=0).repeat(num_grids, axis=2)
    z = z.repeat(num_grids, axis=0).repeat(num_grids, axis=1)

    return x, y, z
