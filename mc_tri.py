#
#    Licensing:
#
#    This code is distributed under the GNU LGPL license.
#

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.append(os.path.join('./'))
from rnd_uniform.uniform import r8vec_uniform_01, r8mat_uniform_01, r8_uniform_01
from rnd_uniform.sample import triangle01_sample, cube01_sample, ball01_sample, annulus_sample
from rnd_uniform.sample import circle01_sample_ergodic, circle01_sample_random
from rnd_uniform.sample import hypercube01_sample, polygon_sample, ellipsoid_sample
from base import plot2d, PlotBase, create_tempnum


class MonteCarlo (plot2d):

    def __init__(self, aspect='equal'):
        plot2d.__init__(self, aspect=aspect)
        self.create_tempdir(-1)

        v0 = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 1.0]])

        c0 = [0, 0]
        c1 = [1, -1]
        r1 = 1.0
        r2 = 2.0
        r3 = 0.5
        r4 = 3.0

        a = np.array([
            [9.0, 6.0, 3.0],
            [6.0, 5.0, 4.0],
            [3.0, 4.0, 9.0]])

        v1 = np.array([1.0, 2.0, 3.0])

        seed = 123456789
        n = 2**5
        while (n <= 2**16):
            print("n={:d}".format(n))
            self.PlotTest(*triangle01_sample(n, seed),
                          title="triangle")
            n = 2 * n

    def PlotTest(self, x, seed, title=None):
        dim, num = x.shape
        titletxt = "{} n={:d}".format(title, num)
        if title == None:
            pngname = create_tempnum(self.tempname, ext=".png")
        else:
            pngname = create_tempnum(self.tempname + "_" + title, ext=".png")

        if dim == 2:
            self.new_2Dfig()
            self.axs.scatter(*x, s=0.5)
            self.axs.set_title(titletxt)
            self.SavePng(pngname)
        else:
            self.new_2Dfig()
            self.contourf_tri(*x, title=titletxt, pngname=pngname)
        plt.close("all")


if (__name__ == '__main__'):
    obj = MonteCarlo()
