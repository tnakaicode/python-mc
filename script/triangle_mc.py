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
from base import plot2d
from rnd_uniform.sample import triangle01_sample
obj = plot2d()


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


def reference_to_physical_t3(t, n, ref):

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


def triangle_monte_carlo_test01():

    # *****************************************************************************80
    #
    # TRIANGLE_MONTE_CARLO_TEST01 samples the unit triangle.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    15 August 2009
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np

    t = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]])

    print('')
    print('TRIANGLE_MONTE_CARLO_TEST01')
    print('  Integrate xy^3')
    print('  Integration region is the unit triangle.')
    print('  Use an increasing number of points N.')

    seed = 123456789

    print('')
    print('     N          XY^3')
    print('')

    n = 1

    while (n <= 65536):
        result, seed = triangle_monte_carlo(t, n, triangle_integrand, seed)
        print('  %8d  %14f' % (n, result))
        n = 2 * n
    return


def triangle_integrand(p):

    # *****************************************************************************80
    #
    # TRIANGLE_INTEGRAND evaluates xy^3
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    15 August 2009
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, real P(2,P_NUM), the evaluation points.
    #
    #    Output, real FP(P_NUM), the integrand values.
    #
    fp = p[0, :] * p[1, :] ** 3

    return fp


def triangle_monte_carlo_test02():

    # *****************************************************************************80
    #
    # TRIANGLE_MONTE_CARLO_TEST02 samples a general triangle.
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

    t = np.array([
        [2.0, 3.0, 0.0],
        [0.0, 4.0, 3.0]])

    print('')
    print('TRIANGLE_MONTE_CARLO_TEST02')
    print('  Integrate xy^3')
    print('  Integration region is a general triangle.')
    print('  Use an increasing number of points N.')

    seed = 123456789

    print('')
    print('     N          XY^3')
    print('')

    n = 1

    while (n <= 65536):

        result, seed = triangle_monte_carlo(t, n, triangle_integrand, seed)

        print('  %8d  %14f' % (n, result))

        n = 2 * n

    return


def triangle_integrand(p):

    # *****************************************************************************80
    #
    # TRIANGLE_INTEGRAND evaluates xy^3
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    15 August 2009
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, real P(2,P_NUM), the evaluation points.
    #
    #    Output, real FP(P_NUM), the integrand values.
    #
    fp = p[0, :] * p[1, :] ** 3

    return fp


if (__name__ == '__main__'):
    obj.create_tempdir(-1)
    triangle_monte_carlo_test()
    # triangle_monte_carlo_test01()
    # triangle_monte_carlo_test02()
