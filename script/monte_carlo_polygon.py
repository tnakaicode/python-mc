#! /usr/bin/env python3
#
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import platform

sys.path.append(os.path.join('../'))
from base import plot2d, plot3d
from rnd_uniform.polygon import polygon_triangulate, polygon_area, triangle_area
from rnd_uniform.uniform import r8_uniform_01, r8vec_uniform_01, r8vec_ergodic
from rnd_uniform.sample import circle01_sample_ergodic
from rnd_uniform.monomial import monomial_value


def polygon_monte_carlo_test():

    #
    # POLYGON_MONTE_CARLO_TEST estimates integrals over a polygon in 2D.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    13 November 2016
    #
    #  Author:
    #
    #    John Burkardt
    #

    nv = 4

    v = np.array([
        [-0.5, -0.5],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0]])

    e_name = ["X", "Y"]
    e_test = np.array([
        [0, 0],
        [2, 0],
        [0, 2],
        [4, 0],
        [2, 2],
        [0, 4],
        [6, 0]])

    print('')
    print('POLYGON_MONTE_CARLO_TEST')
    print('  Python version: %s' % (platform.python_version()))
    print('  Use POLYGON_SAMPLE to estimate integrals')
    print('  over the interior of a polygon in 2D.')

    obj = plot2d()
    obj.create_tempdir(-1)
    seed = 123456789

    print('')
    txt = " \tN"
    for e in e_test:
        txt += "\t"
        for idx, vname in enumerate(e_name):
            txt += "{}^{:d}".format(vname, e[idx])
    print(txt)
    print('')

    n = 2**10
    while (n <= 2**17):
        x, seed = polygon_sample(nv, v, n, seed)
        print('  %8d' % (n), end='')
        for e in e_test:
            value = monomial_value(n, 2, e, x)
            result = polygon_area(
                nv, v[:, 0], v[:, 1]) * np.sum(value[0:n]) / float(n)
            print('\t%14.6g' % (result), end='')
        print('')

        obj.axs.scatter(x[:, 0], x[:, 1], s=0.5)
        obj.axs.set_title("n={:d}".format(n))
        obj.SavePng_Serial()
        plt.close()
        obj.new_fig()

        n = 2 * n

    #print('     Exact'),
    # for e in e_test:
    #    result = polygon_monomial_integral(nv, v, e)
    #    print('  %14.6g' % (result)),
    # print('')

    print('')
    print('POLYGON_MONTE_CARLO_TEST')
    print('  Normal end of execution.')
    return


def polygon_sample(nv, v, n, seed):

    # *****************************************************************************80
    #
    # POLYGON_SAMPLE uniformly samples a polygon.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    18 October 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer NV, the number of vertices.
    #
    #    Input, real V(NV,2), the vertices of the polygon, listed in
    #    counterclockwise order.
    #
    #    Input, integer N, the number of points to create.
    #
    #    Input/output, integer SEED, a seed for the random
    #    number generator.
    #
    #    Output, real S(2,N), the points.
    #
    #  Triangulate the polygon.

    x = np.zeros(nv)
    y = np.zeros(nv)
    for j in range(0, nv):
        x[j] = v[j, 0]
        y[j] = v[j, 1]

    #  Determine the areas of each triangle.
    triangles = polygon_triangulate(nv, x, y)
    area_triangle = np.zeros(nv - 2)

    area_polygon = 0.0
    for i in range(0, nv - 2):
        area_triangle[i] = triangle_area(
            v[triangles[i, 0], 0], v[triangles[i, 0], 1],
            v[triangles[i, 1], 0], v[triangles[i, 1], 1],
            v[triangles[i, 2], 0], v[triangles[i, 2], 1])
        area_polygon = area_polygon + area_triangle[i]

    #  Normalize the areas.
    area_relative = np.zeros(nv - 1)

    for i in range(0, nv - 2):
        area_relative[i] = area_triangle[i] / area_polygon

    #  Replace each area by the sum of itself and all previous ones.
    area_cumulative = np.zeros(nv - 2)

    area_cumulative[0] = area_relative[0]
    for i in range(1, nv - 2):
        area_cumulative[i] = area_relative[i] + area_cumulative[i - 1]

    s = np.zeros([n, 2])

    for j in range(0, n):
        #  Choose triangle I at random, based on areas.
        area_percent, seed = r8_uniform_01(seed)
        i = None
        for k in range(0, nv - 2):
            i = k
            if (area_percent <= area_cumulative[k]):
                break

        #  Now choose a point at random in triangle I.
        r, seed = r8vec_uniform_01(2, seed)
        #r, seed = r8vec_ergodic(2, seed)

        if (1.0 < r[0] + r[1]):
            r[0] = 1.0 - r[0]
            r[1] = 1.0 - r[1]

        s[j, 0] = (1.0 - r[0] - r[1]) * v[triangles[i, 0], 0] \
            + r[0] * v[triangles[i, 1], 0] \
            + r[1] * v[triangles[i, 2], 0]

        s[j, 1] = (1.0 - r[0] - r[1]) * v[triangles[i, 0], 1] \
            + r[0] * v[triangles[i, 1], 1] \
            + r[1] * v[triangles[i, 2], 1]

    return s, seed


def timestamp():
    import time

    t = time.time()
    print(time.ctime(t))

    return None


if (__name__ == '__main__'):
    timestamp()
    polygon_monte_carlo_test()
    timestamp()
