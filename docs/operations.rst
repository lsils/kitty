Operations
==========

The header ``<kitty/operations.hpp>`` implements several operations on truth tables.

Predicates
----------

.. doc_brief_table::
   equal
   less_than
   has_var
   has_var_inplace
   is_const0
   implies
   intersection_is_empty

Combination and manipulation
----------------------------

.. doc_brief_table::
   unary_not
   binary_and
   binary_or
   binary_xor
   ternary_ite
   ternary_majority
   mux_var
   next_inplace
   next
   cofactor0_inplace
   cofactor0
   cofactor1_inplace
   cofactor1
   swap_inplace
   swap
   swap_adjacent_inplace
   swap_adjacent
   flip_inplace
   flip
   min_base_inplace
   expand_inplace
   extend_to_inplace
   extend_to
   shrink_to_inplace
   shrink_to
   shift_left_inplace
   shift_left
   shift_right_inplace
   shift_right
   shift_with_mask_inplace
   shift_with_mask
