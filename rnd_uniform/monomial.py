#! /usr/bin/env python3
#
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.append(os.path.join('../'))


def monomial_value(n, m, e, x):

    #
    # MONOMIAL_VALUE evaluates a monomial.
    #
    #  Discussion:
    #
    #    This routine evaluates a monomial of the form
    #
    #      product ( 1 <= i <= m ) x(i)^e(i)
    #
    #    The combination 0.0^0, if encountered, is treated as 1.0.
    #
    #  Parameters:
    #
    #    Input, integer M, the spatial dimension.
    #
    #    Input, integer N, the number of evaluation points.
    #
    #    Input, integer E(M), the exponents.
    #
    #    Input, real X(N,M), the point coordinates.
    #
    #    Output, real V(N), the monomial values.
    #
    import numpy as np

    v = np.ones(n)

    for i in range(0, m):
        if (0 != e[i]):
            for j in range(0, n):
                v[j] = v[j] * x[j, i] ** e[i]

    return v
