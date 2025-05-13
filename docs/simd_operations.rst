Single Instruction Multiple Data Operations
===========================================

Dynamic truth table
-------------------

The header ``<kitty/simd_operations.hpp>`` implements bitwise operations
that are optimized through vectorization, if supported by the machine on
which the code is compiled. It is suggested to replace the traditional 
operations with these alternatives for truth table with more than `1024`
bits. However, compiler dependent profiling is required to identify the
performance cutoff.

The header provides the following functions:

.. doc_brief_table::
   simd::bitwise_and
   simd::bitwise_or
   simd::bitwise_xor
   simd::bitwise_lt
   simd::unary_not
   simd::set_zero
   simd::set_ones