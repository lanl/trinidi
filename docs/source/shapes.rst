.. _trinidi_shapes:

Array Shapes
============

Measurements
------------

Open beam measurement, ``Y_o``, and the sample measurement, ``Y_s`` both
need to be arrays with the same shapes, ``projection_shape + (N_A,)``:

.. math::    \mathrm{Shape}(Y_\mathrm{o}) = \mathrm{Shape}(Y_\mathrm{s}) &= \texttt{projection_shape + (N_A,)} \\
                &= \texttt{(N_p1, N_p2, ..., N_pn, N_A)}



The tuple ``projection_shape`` describes the dimensions that are `not` the hyperspectral dimensions
and is thus usually equal to the detector shape, i.e. ``(N_y, N_x)``.
The integer ``N_A`` is the number of time-of-arrival (TOA) bins.
The integer ``N_F`` is the number of time-of-flight (TOF) bins.
For more detail see :ref:`resolution_shapes`.




Units
-----
ToDo add units of everyting.





.. _resolution_shapes:

Resolution Operator
-------------------

``N_A`` vs ``N_F``.


+------------------------+----------------+----------------------+
| Description            | Quantity       |  Shape               |
+------------------------+----------------+----------------------+
| Time-of-flight vector  |      ``t_F``   | ``(N_F,)``           |
+------------------------+----------------+----------------------+
| Time-of-arrival vector | ``t_A``        | ``(N_A,)``           |
+------------------------+----------------+----------------------+
| Areal Density [mol/cm²]| ``Z``          | ``N_p x (N_M,)``     |
+------------------------+----------------+----------------------+
| Resolution Operator    | ``R``          | ``N_p x (N_F, N_A)`` |
+------------------------+----------------+----------------------+
|                        | ``D``          | ``N_M x N_F``        |
+------------------------+----------------+----------------------+
