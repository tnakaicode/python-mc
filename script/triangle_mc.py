#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.append(os.path.join('../'))
from base import plot2d, plot3d
from rnd_uniform.sample import triangle01_sample, triangle02_sample


def triangle_monte_carlo_test():

    # *****************************************************************************80
    #
    # TRIANGLE_MONTE_CARLO_TEST estimates integrals over a general triangle.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    18 July 2018
    #
    #  Author:
    #
    #    John Burkardt
    #
    m = 2

    e_test = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [2, 0],
        [1, 1],
        [0, 2],
        [3, 0]])

    print('')
    print('TRIANGLE_MONTE_CARLO_TEST')
    print('  TRIANGLE_MONTE_CARLO estimates an integral over')
    print('  a general triangle using the Monte Carlo method.')

    t1 = np.array([
        [0.0, 3.0, 0.0],
        [0.0, 4.0, 3.0]])
    t2 = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 4.0, 5.0]])

    seed = 123456789

    #area = triangle_area(t)
    obj = plot2d()
    obj.create_tempdir(-1)

    n = 2**5
    while (n <= 2**16):
        p0, seed = triangle01_sample(n, seed)
        p1 = reference_to_physical_t3(t1, n, p0)
        p2 = reference_to_physical_t3(t2, n, p0)

        obj.new_fig()
        obj.axs.scatter(*p0, s=0.5)
        obj.axs.set_title("n={:d}".format(n))
        obj.SavePng_Serial(obj.tempname + "-p0.png")

        obj.new_fig()
        obj.axs.scatter(*p1, s=0.5)
        obj.axs.set_title("n={:d}".format(n))
        obj.SavePng_Serial(obj.tempname + "-p1.png")
        plt.close()

        obj.new_fig()
        obj.axs.scatter(*p2, s=0.5)
        obj.axs.set_title("n={:d}".format(n))
        obj.SavePng_Serial(obj.tempname + "-p2.png")
        plt.close()

        n = 2 * n

    #
    #  Terminate.
    #
    print('')
    print('TRIANGLE_MONTE_CARLO_TEST:')
    print('  Normal end of execution.')
    return


def reference_to_physical_t3(t, n=2 ** 10, ref=np.zeros([2, 2**10])):

    #
    # REFERENCE_TO_PHYSICAL_T3 maps a reference point to a physical point.
    #
    #  Discussion:
    #
    #    Given the vertices of an order 3 physical triangle and a point
    #    (XSI,ETA) in the reference triangle, the routine computes the value
    #    of the corresponding image point (X,Y) in physical space.
    #
    #    Note that this routine may also be appropriate for an order 6
    #    triangle, if the mapping between reference and physical space
    #    is linear.  This implies, in particular, that the sides of the
    #    image triangle are straight and that the "midside" nodes in the
    #    physical triangle are halfway along the sides of
    #    the physical triangle.
    #
    #  Reference Element T3:
    #
    #    |
    #    1  3
    #    |  |\
    #    |  | \
    #    S  |  \
    #    |  |   \
    #    |  |    \
    #    0  1-----2
    #    |
    #    +--0--R--1-->
    #
    #  Parameters:
    #
    #    Input, real T(2,3), the coordinates of the vertices.  The vertices are assumed
    #    to be the images of (0,0), (1,0) and (0,1) respectively.
    #
    #    Input, integer N, the number of points to transform.
    #
    #    Input, real REF(2,N), the coordinates of points in the reference space.
    #
    #    Output, real PHY(2,N), the coordinates of the corresponding points in the
    #    physical space.
    #

    phy = np.zeros([2, n])
    for i in range(0, 2):
        phy[i, :] = t[i, 0] * (1.0 - ref[0, :] - ref[1, :]) + \
            t[i, 1] * ref[0, :] + t[i, 2] * ref[1, :]

    return phy


def reference_to_physical_t4(t, n=2 ** 10, ref=np.zeros([2, 2**10])):
    phy = np.zeros_like(ref)
    for i, val in enumerate(phy[:, 0]):
        phy[i, :] = t[i, 0] * (1.0 - ref[0, :] - ref[1, :]) + \
            t[i, 1] * ref[0, :] + t[i, 2] * ref[1, :]
    return phy


if (__name__ == '__main__'):
    seed = 123456789
    p1, seed = triangle01_sample(3, seed)
    t1 = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 4.0, 5.0]])
    p1 = reference_to_physical_t3(t1, 3, p1)
    print(p1)

    p1, seed = triangle01_sample(3, seed)
    t1 = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 4.0, 5.0]])
    p1 = reference_to_physical_t4(t1, 3, p1)
    print(p1)

    obj = plot3d()
    p2, seed = triangle02_sample(3, seed)
    print(p2)
    t2 = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 4.0, 5.0],
        [1.0, 3.0, 2.0]])
    p2 = reference_to_physical_t4(t2, 3, p2)
    print(p2)
    obj.axs.plot(*p2)
    obj.SavePng()

    triangle_monte_carlo_test()
    # triangle_monte_carlo_test01()
    # triangle_monte_carlo_test02()
