Sets of Pairs of Functions to be Distinguished
==============================================

The header ``<kitty/spfd.hpp>`` implements a data structure for
the manipulation of Sets of Pairs of Functions to be Distinguished ( SPFD )

The class :cpp:class:`kitty::spfd` provides the
following public member functions.

+-----------------------------------+----------------------------------------------------------------------------------+
| Function                          | Description                                                                      |
+===================================+==================================================================================+
| ``spfd()``                        | Standard constructor. Stores the SPFD for the constant 0 function                |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``spfd(onset)``                   | Constructor from the onset of a completely specified function.                   |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``spfd(onset, careset)``          | Constructor from the onset and careset of an incompletely specified function.    |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``reset()``                       | Restores the SPFD to the configuration at construction time.                     |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``reset(onset)``                  | Re-initializes the SPFD to the one of a completely specified function.           |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``reset(onset, careset)``         | Re-initializes the SPFD to the one of an incompletely specified function.        |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``update(onset_other)``           | Removes the minterm pairs distinguished by a completely specified function.      |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``evaluate(onset_other)``         | Counts the minterm pairs to be distinguished after ``update(onset_other)``.      |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``is_covered()``                  | Check if there are no remaining minterm pairs to be distinguished.               |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``is_saturated()``                | Check if the number of sets exceeds the memory limit reserved at compile time.   |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``get_num_pairs()``               | Getter for the number of minterm pairs top be distinguished.                     |
+-----------------------------------+----------------------------------------------------------------------------------+
| ``count_pairs()``                 | Explicitly recomputes the number of minterm pairs to be distinguished.           |
+-----------------------------------+----------------------------------------------------------------------------------+

.. doxygenstruct:: kitty::spfd
   :members:
