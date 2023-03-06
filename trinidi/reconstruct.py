"""Some reconstruction functions and classes."""


class Parameters:
    r"""Parameter class for nuisance parameters.


    `projection_shape` is the shape of the detector so usually this will
        be (N_pixels_x, N_pixels_y) but it may be any shape including
        singleton.

    `N_p` number of projections, `np.prod(projection_shape) = N_p`.

    `Y_o`, `Y_s` measurements have shape `projection_shape +  (N_A,)`

    `N_A` is the number of measured TOF bins (TOA's)

    'D' has shape `(N_F, N_A)`

    `N_F` is the number of theoretical TOF bins. :math:`N_F \geq N_A`

    `ω_sz`, `ω_s0` have shape `projection_shape + (1,)`.
        :math:`ω_sz^\top` has shape `(1,) + projection_shape`.

    `R` has shape `(N_F, N_A)`.



    """

    def __init__(self, Y_o, Y_s, D, ω_sz, ω_s0=None, R=None, β=1.0):
        r"""
        Args:
            Y_o: Open beam measurement.
            Y_s: Sample measurement.
            D: Cross section dictionary.
            ω_sz: Uniformly dense region averaging vector.
            ω_s0: Open beam region averaging vector. When `None`,
                parameters will be computed equivalent to `β=0`.
            R: Resolution operator of class `ResolutionOperator`.
                When `None`, `R` is chosen internally as identity
                operator.
            β: Balancing weight between solving equation for `ω_sz`
                (`β=0`), and solving equation for `ω_s0` (`β` infinite).
                Equal weight when `β=1.0` (default).
        """
