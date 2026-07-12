Energy Assembly
===============

The Kohn-Sham total energy:

.. math::
   E_{\text{tot}} = E_{\text{kin}} + E_{\text{ne}} + E_H
       + E_{xc} + E_{\text{ion}}

where:

* :math:`E_{\text{kin}} = \sum_k f_k \epsilon_k
      - \text{Tr}(P (V_{\text{ext}} + V_H + V_{xc}))`
* :math:`E_{\text{ne}} = \text{Tr}(P V_{\text{ext}})`
* :math:`E_H = \frac{1}{2}\text{Tr}(P V_H)`
* :math:`E_{xc} = \text{Tr}(P \epsilon_{xc})` (energy density, not potential)
* :math:`E_{\text{ion}} = \frac{1}{2}\sum_{I \neq J} Z_I Z_J / |R_I - R_J|`
  (Ewald for periodic)

All traces use FP64-emulated reductions (f64e) on the production path.
