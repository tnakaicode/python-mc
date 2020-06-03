import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from scipy.spatial import ConvexHull, Delaunay
from optparse import OptionParser

sys.path.append(os.path.join("./"))
from base import plot2d, plot3d

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)


class ConvexArea3D (plot3d):

    def __init__(self, num=30, idx=2):
        plot3d.__init__(self)
        self.pnt = np.random.rand(num, 3)
        self.cov = ConvexHull(self.pnt)
        self.axs.plot(self.pnt[:, 0], self.pnt[:, 1], self.pnt[:, 2], 'o')

        print(self.cov)
        for idx in self.cov.simplices:
            x = self.pnt[idx, 0]
            y = self.pnt[idx, 1]
            z = self.pnt[idx, 2]
            print(idx, x, y, z)
            self.axs.plot_trisurf(x, y, z, linewidth=0.2, alpha=0.3)

        print(self.cov.vertices)

        self.axs.plot(self.pnt[self.cov.vertices, 0],
                      self.pnt[self.cov.vertices, 1], 'r--', lw=2)


class ConvexArea2D (plot2d):

    def __init__(self, num=30, idx=2):
        plot2d.__init__(self)
        self.pnt = np.random.rand(num, 2)
        self.cov = ConvexHull(self.pnt)
        self.tri = Delaunay(self.pnt)
        self.axs.plot(self.pnt[:, 0], self.pnt[:, 1], 'o')

        print(self.cov)
        for idx in self.cov.simplices:
            x = self.pnt[idx, 0]
            y = self.pnt[idx, 1]
            print(idx, x, y)
            self.axs.plot(x, y, 'r--', lw=1.0)
        print(self.cov.vertices)

        for idx in self.tri.simplices:
            x = self.pnt[idx, 0]
            y = self.pnt[idx, 1]
            print(idx, x, y)
            self.axs.plot(x, y, 'b--', lw=0.2)


if __name__ == '__main__':
    argvs = sys.argv
    parser = OptionParser()
    parser.add_option("--dir", dest="dir", default="./")
    parser.add_option("--pxyz", dest="pxyz",
                      default=[0.0, 0.0, 0.0], type="float", nargs=3)
    opt, argc = parser.parse_args(argvs)
    print(opt, argc)

    px = np.linspace(-1, 1, 100) * 100 + 50
    py = np.linspace(-1, 1, 200) * 100 - 50
    mesh = np.meshgrid(px, py)

    obj = ConvexArea2D(num=100)
    obj.SavePng()
