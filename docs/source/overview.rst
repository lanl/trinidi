.. _overview:

Overview
========

The main functionality of TRINIDI is reconstructing areal density
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

In TRINIDI the measurements are handled using the
``nd-arrays`` ``Y_s`` and ``Y_o`` and
::

        Y_s.shape == Y_o.shape == projection_shape + (N_A,).

The tuple ``projection_shape`` is typically the shape of the detector and
``np.prod(projection_shape) == N_P``. For example, given a measurement
with a :math:`64 \times 64` pixel detector and :math:`1000` TOA bins we
get
::

        Y_s.shape == projection_shape + (N_A,) == (64, 64) + (1000,) == (64, 64, 1000).

The array element ``Y_s[i_y, i_x, i_t]`` corresponds to the pixel with
index ``(i_x, i_y)`` and TOA bin ``t_A[i_t]``, where ``t_A`` has size
``N_A``.

Similarly, for the areal density array we have
::

        Z.shape == projection_shape + (N_m,).

Below you can find a :ref:`summary <shape_summary>` of the most
important arrays and operators in TRINIDI.


Cross Section Dictionary
------------------------

The cross section dictionary,
:math:`D \in \mathbb{R}^{N_{\mathrm{m}} \times N_{\mathrm{F}}}`, maps
areal densities to hyperspectral attenuation or transmission values.
Ignoring the effects of the resolution function, the transmission is
expressed as

.. math:: Q_0 = \exp(-ZD)

so that :math:`(Q_0)_{i, j}` is the fraction of flux neutrons that make it
to the detector at projection :math:`i` and TOF bin index :math:`j`.

In TRINIDI, the cross section data structure, ``D``, is created
using the :class:`.XSDict` class in the :mod:`trinidi.cross_section`
module. This class has features including reading, merging and plotting
cross section entries. The actual values of :math:`D` are stored in
``D.values`` so that the expression of the transmission from above
can be computed as

::

        Q_0 = np.exp(- Z @ D.values).

In TRINIDI, ``D.values`` has units of
:math:`\mathrm{mol}/\mathrm{cm}^2`, which is the reciprocal of the
units of ``Z``.

To make yourself familiar of the cross section dictionary data structure
we recommend to check out the
:ref:`trinidi.cross_section Module Demo <cross_section_demo>`
script which is also available as
:ref:`Jupyter notebook <examples_notebooks>`.



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


Resolution Operator
-------------------

To make yourself familiar of the resolution function data structure
we recommend to check out the
:ref:`trinidi.resolution Module Demo <resolution_demo>`
script which is also available as
:ref:`Jupyter notebook <examples_notebooks>`.


Reconstruction
--------------

To make yourself familiar of the reconstruction module
we recommend to check out the
:ref:`trinidi.reconstruct Module Demo <reconstruct_demo>`
script which is also available as
:ref:`Jupyter notebook <examples_notebooks>`.


.. _shape_summary:

Summary of Array and Operator Shapes and Units
----------------------------------------------

::

    Data Structure    |  Shape
    ------------------+--------------
    t_F               |  (N_F,)
    t_A               |  (N_A,)
    R                 |  (N_F, N_A) (implied)
    D.values          |  (N_m, N_F)
    Z                 |  projection_shape + (N_m,)
    Y_o, Y_s, Φ, B    |  projection_shape + (N_A,)

::

    Quantity          |  Unit       |  Data Structure
    ------------------+-------------+--------------------
    lengths           |  m          |  flight_path_lenght
    times             |  μs         |  t_A, t_F, Δt, t_0
    neutron energies  |  eV         |  E
    cross sections    |  mol/cm²    |  D
    areal densities   |  cm²/mol    |  Z
