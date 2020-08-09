#! /usr/bin/env python
#
import numpy as np
import platform


def r8_normal_01(seed):

    # *****************************************************************************80
    #
    # R8_NORMAL_01 samples the standard normal probability distribution.
    #
    #  Discussion:
    #
    #    The standard normal probability distribution function (PDF) has
    #    mean 0 and standard deviation 1.
    #
    #    The Box-Muller method is used.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    21 January 2016
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer SEED, a seed for the random number generator.
    #
    #    Output, real X, a sample of the standard normal PDF.
    #
    #    Output, integer SEED, an updated seed for the random number generator.
    #

    r1, seed = r8_uniform_01(seed)
    r2, seed = r8_uniform_01(seed)

    x = np.sqrt(- 2.0 * np.log(r1)) * np.cos(2.0 * np.pi * r2)

    return x, seed


def r8vec_normal_01(n, seed):

    # *****************************************************************************80
    #
    # R8VEC_NORMAL_01 returns a unit pseudonormal R8VEC.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    04 March 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer N, the number of entries in the vector.
    #
    #    Input, integer SEED, a seed for the random number generator.
    #
    #    Output, real X(N), the vector of pseudorandom values.
    #
    #    Output, integer SEED, an updated seed for the random number generator.
    #

    x = np.zeros(n)

    for i in range(0, n):
        x[i], seed = r8_normal_01(seed)

    return x, seed


def r8vec_normal_01_test():

    # *****************************************************************************80
    #
    # R8VEC_NORMAL_01_TEST tests R8VEC_NORMAL_01.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    04 March 2015
    #
    #  Author:
    #
    #    John Burkardt
    #

    n = 10
    seed = 123456789

    print('')
    print('R8VEC_NORMAL_01_TEST')
    print('  Python version: %s' % (platform.python_version()))
    print('  R8VEC_NORMAL_01 returns a vector of Normal 01 values')
    print('')
    print('  SEED = %d' % (seed))

    r, seed = r8vec_normal_01(n, seed)

    r8vec_print(n, r, '  Vector:')
#
#  Terminate.
#
    print('')
    print('R8VEC_NORMAL_01_TEST:')
    print('  Normal end of execution.')
    return
