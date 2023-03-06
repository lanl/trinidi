.. _trinidi_shapes:

Array Shapes
============

Measurements
------------

Open beam measurement, ``Y_o``, and the sample measurement, ``Y_s`` both
need to be arrays with the same shapes:

:math:`\mathrm{Shape}(Y_o)` = :math:`\mathrm{Shape}(Y_s)` = ``projection_shape + (N_A,)``


.. math::    \mathrm{Shape}(Y_o) = \mathrm{Shape}(Y_s) = \mathrm{projection_shape}


    ::

        Y_o.shape == Y_s.shape == projection_shape + (N_A,).

The tuple ``projection_shape`` is usually equal to the number of pixels in y-direction and
x-direction, i.e. ``(N_y, N_x)``. The integer ``N_A`` is the number of time-of-arrival (TOA) bins.
The integer ``N_F`` is the number of time-of-flight (TOF) bins.
For more detail see :ref:`resolution_shapes`.










.. _resolution_shapes:

Resolution Operator
-------------------

``N_A`` vs ``N_F``.
