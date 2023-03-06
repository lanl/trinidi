.. _trinidi_shapes:


Measurements
------------

Open beam measurement, ``Y_o``, and the sample measurement, ``Y_s`` both
need to be arrays with the same shapes:

    ::
        projection_shape + (N_A,).

The tuple ``projection_shape`` describes the shape of the detector and
thus is usually equal to the pixels in y-direction and x-direction,
``(N_y, N_x)``. The integer ``N_A`` is the number of `measured`
time-of-flight (TOF) bins, more precisely time-of-arrival (TOA) bins.
(See :ref:`resolution_shapes`.)










.. _resolution_shapes:

Resolution Operator
-------------------

``N_A`` vs ``N_F``.
