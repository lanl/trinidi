.. _overview:

Overview
========

The main functionality of ``TRINIDI`` is reconstructing areal density
values, :math:`Z \in \mathbb{R}^{N_{\mathrm{P}} \times N_{\mathrm{m}}}`,
from the neutron time-of-flight transmission measurements,
:math:`Y_{\mathrm{s}} \in \mathbb{R}^{N_{\mathrm{P}} \times N_{\mathrm{A}}}`
(sample measurement) and
:math:`Y_{\mathrm{o}} \in \mathbb{R}^{N_{\mathrm{P}} \times N_{\mathrm{A}}}`
(open beam measurement), where

.. math:: N_{\mathrm{P}} &= \text{Number of projections} \\
    N_{\mathrm{A}} &= \text{Number of measured time bins} \\
    N_{\mathrm{m}} &= \text{Number of isotopes}

We strongly advise the reader to review our manuscript
:cite:`balke2023trinidi` make yourself familiar with

- assumptions made of the measurement system
- time-of-flight (TOF) and how it relates to the neutron energy
- resolution function, :math:`R`, and how it relates the time-of-flight (TOF) and the time-of-arrival (TOA)
- TOF vector, :math:`t_F`, and TOA vector, :math:`t_A`.
- cross section dictionary, :math:`D`

Array Shapes
------------

In ``TRINIDI`` the measurements are handled using the
``nd-arrays`` ``Y_s`` and ``Y_o`` and
::

        Y_s.shape == Y_o.shape == projection_shape + (N_A,).

The tuple ``projection_shape`` is typically the shape of the detector and
``np.prod(projection_shape) == N_P``. For example, given a measurement
with a :math:`64 \times 64` pixel detector and :math:`1000` TOA bins we
get
::

        Y_s.shape == projection_shape + (N_A,) == (64, 64) + (1000,) == (64, 64, 1000)

The array element ``Y_s[i_y, i_x, i_t]`` corresponds to the pixel with
index ``(i_x, i_y)`` and TOA bin ``t_A[i_t]``, where ``t_A`` has size
``N_A``.

Similarly, for the areal density array we have
::

        Z.shape == projection_shape + (N_m,).


Time-of-Flight Calibration
--------------------------

In practice, the first step will usually be to estimate the ``t_A``
array. We assume the TOF (TOA) bins are equidistant and the time vector
``t_F`` (``t_A``) is in increasing order. Mostly, we leave the calibration
to the liberty of the user, however we provide an example illustration
that we recommend you read first. For this check out the
:ref:`Time-Energy Calibration Demo <time_energy_calibration_demo>`
script which is also available as
:ref:`Jupyter notebook <examples_notebooks>`.




Units Summary
-------------

::

    Quantity          |  Unit       |  Data Structure
    ------------------+-------------+--------------------
    lengths           |  m          |  flight_path_lenght
    times             |  μs         |  t_A, t_F, Δt, t_0
    neutron energies  |  eV         |  E
    cross sections    |  mol/cm²    |  D
    areal densities   |  cm²/mol    |  Z



Shapes Summary
--------------

::

    Data Structure    |  Shape
    ------------------+--------------
    t_F               |  (N_F,)
    t_A               |  (N_A,)
    R                 |  (N_F, N_A) (implied)
    D.values          |  (N_m, N_F)
    Z                 |  (N_P, N_m)
    Y_o, Y_s          |  (N_P, N_A)
