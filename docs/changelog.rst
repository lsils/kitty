Change Log
==========

Next release
------------

v0.8 (September 9, 2022)
------------------------

* Data structure: ``spfd`` (contributed by Andrea Costamagna)
  `#140 <https://github.com/msoeken/kitty/pull/140>`_

* Data structure: ``ternary_truth_table`` and ``quaternary_truth_table`` (contributed by Siang-Yun Lee and Gianluca Radi)
  `#133 <https://github.com/msoeken/kitty/pull/133>`_

* Print: ``print_kmap`` (contributed by Gianluca Radi)
  `#129 <https://github.com/msoeken/kitty/pull/129>`_

* Constructors: ``create_from_formula`` (contributed by Alessandro Tempia Calvino)
  `#123 <https://github.com/msoeken/kitty/pull/123>`_

* Canonization: ``exact_n_canonization``, ``exact_n_canonization_complete``, ``exact_np_enumeration``, ``exact_p_enumeration``, ``exact_n_enumeration`` (contributed by Alessandro Tempia Calvino)
  `#123 <https://github.com/msoeken/kitty/pull/123>`_

* Data structure: ``partial_truth_table`` (contributed by Siang-Yun Lee)
  `#102 <https://github.com/msoeken/kitty/pull/102>`_

* Properties: ``is_covered_with_divisors`` (contributed by Heinz Riener)
  `#106 <https://github.com/msoeken/kitty/pull/106>`_

* Algorithm: ``ternary_predicate`` (contributed by Siang-Yun Lee)
  `#120 <https://github.com/msoeken/kitty/pull/120>`_

* Operation: ``intersection_is_empty`` (contributed by Siang-Yun Lee)
  `#120 <https://github.com/msoeken/kitty/pull/120>`_

v0.7 (March 13, 2020)
---------------------

* Properties: ``is_normal``, ``is_trivial`` (contributed by Winston Haaswijk)
  `#75 <https://github.com/msoeken/kitty/pull/75>`_

* Operations: ``shift_with_mask_inplace`` and ``shift_with_mask``
  `#80 <https://github.com/msoeken/kitty/pull/80>`_

* Compute simple SPP from ESOP form
  `#83 <https://github.com/msoeken/kitty/pull/83>`_

* Bit operation: ``count_zeros`` (contributed by Heinz Riener)
  `#84 <https://github.com/msoeken/kitty/pull/84>`_

* Properties: ``absolute_distinguishing_power``, ``relative_distinguishing_power`` (contributed by Heinz Riener)
  `#84 <https://github.com/msoeken/kitty/pull/84>`_

* Properties: ``polynomial_degree``
  `#86 <https://github.com/msoeken/kitty/pull/86>`_

* Canonization: ``spectral_representative``
  `#89 <https://github.com/msoeken/kitty/pull/89>`_

* ANF as expression: ``anf_to_expression``
  `#91 <https://github.com/msoeken/kitty/pull/91>`_

* Constructor: ``nth_var``
  `#95 <https://github.com/msoeken/kitty/pull/95>`_

* Spectral classification: ``hybrid_exact_spectral_canonization``
  `#96 <https://github.com/msoeken/kitty/pull/96>`_

v0.6 (June 2, 2019)
-------------------

* Print function and functions in christmas tree pattern
  `#62 <https://github.com/msoeken/kitty/pull/62>`_

* Get runlength encoding of truth table bits
  `#63 <https://github.com/msoeken/kitty/pull/63>`_

* Constructors: ``create_from_parity``, ``create_prime``
  `#64 <https://github.com/msoeken/kitty/pull/64>`_
  `#67 <https://github.com/msoeken/kitty/pull/67>`_

* Operations: ``implies`` (contributed by Heinz Riener), ``mux_var``
  `#65 <https://github.com/msoeken/kitty/pull/65>`_
  `#66 <https://github.com/msoeken/kitty/pull/66>`_

* Decomposition (DSD): ``is_top_decomposable``, ``is_bottom_decomposable``
  `#66 <https://github.com/msoeken/kitty/pull/66>`_

* Ashenhurst decomposition: ``is_ashenhurst_decomposable``, ``ashenhurst_decomposition`` (contributed by Mahyar Emami)
  `#70 <https://github.com/msoeken/kitty/pull/70>`_

* Bi-decomposition: ``is_bi_decomposable`` (contributed by Eleonora Testa)
  `#72 <https://github.com/msoeken/kitty/pull/72>`_

* Canonization: ``exact_p_canonization`` (contributed by Eleonora Testa)
  `#68 <https://github.com/msoeken/kitty/pull/68>`_

* Properties: ``is_monotone``, ``is_selfdual`` (contributed by Eleonora Testa)
  `#69 <https://github.com/msoeken/kitty/pull/69>`_

v0.5 (October 8, 2018)
----------------------

* Cube methods: ``difference``, ``flip_bit``, ``flip_mask`` (contributed by Heinz Riener)
  `#52 <https://github.com/msoeken/kitty/pull/52>`_
  `#60 <https://github.com/msoeken/kitty/pull/60>`_

* Type traits: ``is_truth_table``
  `#53 <https://github.com/msoeken/kitty/pull/53>`_

* Assignment operator for truth tables
  `#54 <https://github.com/msoeken/kitty/pull/54>`_

* Function properties: ``is_symmetric_in``
  `#55 <https://github.com/msoeken/kitty/pull/55>`_

* Bit operations: ``flip_bit``
  `#56 <https://github.com/msoeken/kitty/pull/56>`_

* Constructors: ``create_from_expression``
  `#58 <https://github.com/msoeken/kitty/pull/58>`_

* Marked ``esop_from_pprm`` and ``esop_from_optimum_pkrm`` as deprecated
  `#59 <https://github.com/msoeken/kitty/pull/59>`_

* Function representative enumeration
  `#61 <https://github.com/msoeken/kitty/pull/61>`_

v0.4 (May 4, 2018)
------------------

* Constructors: ``create``, ``create_from_clauses``, ``create_characteristic``, ``create_multiple_from_chain``
  `#41 <https://github.com/msoeken/kitty/pull/41>`_
  `#43 <https://github.com/msoeken/kitty/pull/43>`_
  `#45 <https://github.com/msoeken/kitty/pull/45>`_

* Operations: ``shrink_to`` and ``shrink_to_inplace``
  `#42 <https://github.com/msoeken/kitty/pull/42>`_
  `#47 <https://github.com/msoeken/kitty/pull/47>`_

* Operation ``extend_to`` is now called ``extend_to_inplace``, new special out-of-place versions ``extend_to``
  `#47 <https://github.com/msoeken/kitty/pull/47>`_

* CNF generation: ``cnf_characteristic``
  `#41 <https://github.com/msoeken/kitty/pull/41>`_

* Print all cubes: ``print_cubes``
  `#44 <https://github.com/msoeken/kitty/pull/44>`_

* Generate implicants and prime implicants: ``get_minterms``, ``get_jbuddies``, ``get_prime_implicants_morreale``
  `#46 <https://github.com/msoeken/kitty/pull/46>`_

* Function properties: ``chow_parameters``, ``is_canalizing``, ``is_horn``, ``is_krom``
  `#48 <https://github.com/msoeken/kitty/pull/48>`_

* Spectral canonization with resource limit
  `#49 <https://github.com/msoeken/kitty/pull/49>`_

* Compute different spectra: ``rademacher_walsh_spectrum``, ``autocorrelation_spectrum``, ``spectrum_distribution``
  `#49 <https://github.com/msoeken/kitty/pull/49>`_

* Compute spectral equivalence class index: ``get_spectral_class``
  `#50 <https://github.com/msoeken/kitty/pull/50>`_

v0.3 (February 25, 2018)
------------------------

* Constructors: ``create_from_chain``, ``create_symmetric``
  `#28 <https://github.com/msoeken/kitty/pull/28>`_
  `#35 <https://github.com/msoeken/kitty/pull/35>`_

* Cube constructors: ``pos_cube``, ``neg_cube``, constructor to create cube from string
  `#29 <https://github.com/msoeken/kitty/pull/29>`_
  `#31 <https://github.com/msoeken/kitty/pull/31>`_

* Cube methods: ``get_bit``, ``get_mask``, ``set_bit``, ``set_mask``, ``clear_bit``, ``clear_mask`` (contributed by Heinz Riener)
  `#30 <https://github.com/msoeken/kitty/pull/30>`_

* Operations: ``shift_left``, ``shift_left_inplace``, ``shift_right``, ``shift_right_inplace``
  `#37 <https://github.com/msoeken/kitty/pull/37>`_
  `#39 <https://github.com/msoeken/kitty/pull/39>`_

* Operators: ``!=``, ``<<``, ``<<=``, ``>>``, ``>>=``
  `#32 <https://github.com/msoeken/kitty/pull/32>`_
  `#37 <https://github.com/msoeken/kitty/pull/37>`_
  `#39 <https://github.com/msoeken/kitty/pull/39>`_

* Return truth table as string: ``to_binary``, ``to_hex``
  `#33 <https://github.com/msoeken/kitty/pull/33>`_

* Default comparison for cubes (contributed by Heinz Riener)
  `#34 <https://github.com/msoeken/kitty/pull/34>`_

* Linear and affine canonization: ``exact_linear_canonization``, ``exact_linear_output_canonization``, ``exact_affine_canonization``, ``exact_affine_output_canonization``
  `#36 <https://github.com/msoeken/kitty/pull/36>`_

* Compute PPRM ESOP for truth table: ``esop_from_pprm``
  `#38 <https://github.com/msoeken/kitty/pull/38>`_

* Compute permutation masks and delta-swap operations: ``delta_swap_inplace``, ``delta_swap``, ``permute_with_masks_inplace``, ``permute_with_masks``, ``compute_permutation_masks``, 
  `#40 <https://github.com/msoeken/kitty/pull/40>`_

v0.2 (December 21, 2017)
------------------------

* Generic algorithms: ``for_each_block``, ``for_each_block_reversed``, ``for_each_one_bit``
  `#14 <https://github.com/msoeken/kitty/pull/14>`_
  `#16 <https://github.com/msoeken/kitty/pull/16>`_

* Constructors: ``create_from_words``, ``create_from_raw``, ``create_threshold``, ``create_equals``
  `#15 <https://github.com/msoeken/kitty/pull/15>`_
  `#20 <https://github.com/msoeken/kitty/pull/20>`_
  `#24 <https://github.com/msoeken/kitty/pull/24>`_
  `#25 <https://github.com/msoeken/kitty/pull/25>`_

* Moved generic operations from `operations.hpp` to `algorithm.hpp`
  `#14 <https://github.com/msoeken/kitty/pull/14>`_

* Print truth tables as string: ``print_binary``, ``print_raw``, ``print_hex``
  `#14 <https://github.com/msoeken/kitty/pull/14>`_
  `#20 <https://github.com/msoeken/kitty/pull/20>`_
  `#22 <https://github.com/msoeken/kitty/pull/22>`_

* Added iterators to all truth tables: ``begin``, ``end``, ``rbegin``, ``rend``, ``cbegin``, ``cend``, ``crbegin``, ``crend``
  `#14 <https://github.com/msoeken/kitty/pull/14>`_

* Count number of ones in truth table: ``count_ones``
  `#21 <https://github.com/msoeken/kitty/pull/21>`_

* Find bits: ``find_first_one_bit``, ``find_last_one_bit``, ``find_first_bit_difference``, ``find_last_bit_difference``
  `#17 <https://github.com/msoeken/kitty/pull/17>`_

* Extend smaller truth tables to larger ones: ``extend_to``
  `#26 <https://github.com/msoeken/kitty/pull/26>`_

* Compute ESOP expressions: ``esop_from_optimum_pkrm``
  `#23 <https://github.com/msoeken/kitty/pull/23>`_

* New data structure for cubes used by ISOP and ESOP computation: ``cube``
  `#23 <https://github.com/msoeken/kitty/pull/23>`_

* NPN canonization heuristics: ``flip_swap_npn_canonization``, ``sifting_npn_canonization``
  `#27 <https://github.com/msoeken/kitty/pull/27>`_

* Spectral canonization: ``exact_spectral_canonization``, ``print_spectrum``
  `#19 <https://github.com/msoeken/kitty/pull/19>`_

v0.1 (October 22, 2017)
-----------------------

* Initial release
* Data structures ``static_truth_table`` and ``dynamic_truth_table``
  `#1 <https://github.com/msoeken/kitty/pull/1>`_

* Bit functions: ``set_bit``, ``get_bit``, ``clear_bit``, ``clear``
  `#1 <https://github.com/msoeken/kitty/pull/1>`_
  `#8 <https://github.com/msoeken/kitty/pull/8>`_

* Constructors: ``create_nth_var``, ``create_from_binary_string``, ``create_from_hex_string``, ``create_random``, ``create_from_cubes``, and ``create_majority``
  `#1 <https://github.com/msoeken/kitty/pull/1>`_
  `#4 <https://github.com/msoeken/kitty/pull/4>`_
  `#5 <https://github.com/msoeken/kitty/pull/5>`_
  `#9 <https://github.com/msoeken/kitty/pull/9>`_
  `#11 <https://github.com/msoeken/kitty/pull/11>`_

* Unary and binary operations: ``unary_not``, ``unary_not_if``, ``binary_and``, ``binary_or``, and ``binary_xor``
  `#2 <https://github.com/msoeken/kitty/pull/2>`_
  `#8 <https://github.com/msoeken/kitty/pull/8>`_

* Ternary operations: ``ternary_majority`` and ``ternary_ite``
  `#3 <https://github.com/msoeken/kitty/pull/3>`_

* Binary predicates: ``equal``, ``less_than``
  `#4 <https://github.com/msoeken/kitty/pull/4>`_
  `#8 <https://github.com/msoeken/kitty/pull/8>`_

* Predicates: ``has_var``, ``is_const0``
  `#11 <https://github.com/msoeken/kitty/pull/11>`_

* Operators: ``~``, ``&``, ``&=``, ``|``, ``|=``, ``^``, ``^=``, ``==``, ``<``
  `#8 <https://github.com/msoeken/kitty/pull/8>`_
  `#11 <https://github.com/msoeken/kitty/pull/11>`_

* Swap adjacent variables: ``swap_adjacent_inplace``, ``swap_adjacent``
  `#6 <https://github.com/msoeken/kitty/pull/6>`_

* Swap variables: ``swap_inplace``, ``swap``
  `#8 <https://github.com/msoeken/kitty/pull/8>`_

* Flip variable: ``flip_inplace``, ``flip``
  `#7 <https://github.com/msoeken/kitty/pull/7>`_

* Enumerate truth tables: ``next_inplace``, ``next``
  `#10 <https://github.com/msoeken/kitty/pull/10>`_

* Compute co-factors: ``cofactor0_inplace``, ``cofactor0``, ``cofactor1_inplace``, ``cofactor1``
  `#11 <https://github.com/msoeken/kitty/pull/11>`_

* Compute minimum base based on functional support: ``min_base_inplace``, ``expand_inplace``
  `#12 <https://github.com/msoeken/kitty/pull/12>`_

* Compute hash values for truth table
  `#13 <https://github.com/msoeken/kitty/pull/13>`_

* NPN canonization: ``exact_npn_canonization``, ``create_from_npn_config``
  `#8 <https://github.com/msoeken/kitty/pull/8>`_

* Compute ISOP representation: ``isop``
  `#11 <https://github.com/msoeken/kitty/pull/11>`_
