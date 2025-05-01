/* kitty: C++ truth table library
 * Copyright (C) 2017-2022  EPFL
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/*!
  \file spfd.hpp
  \brief Implements sets of pairs of functions to be distinguished

  \author Andrea Costamagna
*/

#pragma once

#include <array>

#include "constructors.hpp"

namespace kitty
{

/*! \brief A data structure for Sets of Pairs of Functions to be Distinguished (SPFD).

  This class represents and manipulates subsets of truth tables that define
  the minterm pairs distinguished by a Boolean function. It supports dynamic
  updates through refinement masks and evaluates candidate functions based on
  how well they separate the given function pairs.
 */
template<class TT, uint32_t CapLog2 = 0u>
class spfd
{

#if defined( __clang__ ) || defined( __GNUC__ )
#define PRAGMA_UNROLL _Pragma( "GCC unroll 8" )
#elif defined( _MSC_VER )
#define PRAGMA_UNROLL __pragma( loop( ivdep ) )
#else
#define PRAGMA_UNROLL
#endif

public:
  static constexpr uint32_t capacity = 1u << CapLog2;

public:
  /*! \brief Default constructor.

    When no information is provided, we initialize the data structure to the
    constant 0 function.
  */
  spfd()
  {
    _on_set ^= _on_set;
    _off_set ^= ~_off_set;
    _care_set ^= ~_care_set;
    reset();
  }

  /*! \brief Constructor from onset and careset.

    Save the careset and store the onset and offset.
    Then initialize the class attributes by calling the reset() function

    \param onset Truth table whose 1s identify the onset minterms
    \param careset Truth table whose 1s identify the careset minterms
  */
  spfd( TT const& onset, TT const& careset )
  {
    _care_set = careset;
    _on_set = onset;
    _off_set = ~onset;
    reset();
  }

  /*! \brief Constructor from onset : completely specified function.

    Store the onset and offset and set the careset to a constant 1.
    Then initialize the class attributes by calling the reset() function

    \param onset Truth table whose 1s identify the onset minterms
  */
  spfd( TT const& onset )
  {
    _on_set = onset;
    _off_set = ~onset;
    _care_set = onset ^ ~onset;
    reset();
  }

private:
  /* Delete the copy constructor

    Copies could be expensive or buggy given the internal array
  */
  spfd( const spfd& ) = delete;

  /* Remove the assignment operator

    Copies could be expensive or buggy given the internal array
  */
  spfd& operator=( const spfd& ) = delete;

public:
  /*! \brief Reset the class attributes based on the stored information.

    Restores the initial masks and reset the number of masks and of valid
    masks to the initial value.
  */
  inline void reset()
  {
    _masks[0] = _care_set;
    _num_masks = 1;
    _valid[0] = true;
    _num_valid = 1;
    _num_pairs = count_pairs();
    if ( _num_pairs == 0 )
    {
      _valid[0] = false;
      _num_valid = 0;
    }
  }

  /*! \brief Reset the class attributes after updating the onset.

    Updates the onset information and initializes the masks.
    Since no careset information is given, the method assumes that
    the function is completely specified.

    \param onset new onset to be replaced in the SPFD
  */
  inline void reset( TT const& onset )
  {
    _on_set = onset & _care_set;
    _off_set = ~onset & _care_set;
    _care_set ^= ~_care_set;

    reset();
  }

  /*! \brief Reset the class attributes after updating the onset and careset

    Updates the onset and careset information and initializes the masks.

    \param onset new onset to be replaced in the SPFD
    \param careset new careset to be replaced in the SPFD
  */
  inline void reset( TT const& onset, TT const& careset )
  {
    _care_set = careset;
    _off_set = ~onset & _care_set;
    _on_set = onset & _care_set;
    reset();
  }

  /*! \brief Remove minterm pairs distinguished by a completely specified function

    Each Boolean function distinguishes its onset minterms from its offset minterms.
    This function removes the minterms distinguished by a completely specified function,
    specified by its onset. This is performed by duplicating the masks

    \param onset_other Encodes the minterm pairs to be removed from the SPFD.
  */
  [[nodiscard]] bool update( TT const& onset_other )
  {
    if ( is_saturated() || ( 2 * _num_masks > capacity ) )
    {
      return false;
    }

    _num_pairs = 0;
    uint32_t old_num_masks = _num_masks;

    PRAGMA_UNROLL
    for ( uint32_t i{ 0 }; i < old_num_masks; ++i )
    {
      if ( is_valid( i ) )
      {
        uint32_t const j = _num_masks++;
        _valid[j] = _valid[i];
        _masks[j] = _masks[i] & onset_other;
        _masks[i] &= ~onset_other;
        _valid[j] = !is_constant( j );
        _valid[i] = !is_constant( i );
        assert( ( _valid[i] || ( _num_valid > 0 ) ) );
        _num_valid += _valid[i] ? 1 : -1;
        _num_valid += _valid[j] ? 1 : 0;
        _num_pairs += count_pairs( i );
        _num_pairs += count_pairs( j );
      }
    }
    return true;
  }

public:
  /*! \brief Check if all minterms are distinguished

    The SPFD is empty if:
    - `_num_valid` is 0, i.e., there is no pair of functions to be distinguished.
    - `_num_masks` is non-zero, i.e., there was at least one function initially.
    Set to nodiscard since this check should be used in the code to condition
    subsequent manipulations of the SPFD.
  */
  [[nodiscard]] inline bool is_covered() const
  {
    bool const none_valid = _num_valid == 0;
    bool const many_masks = _num_masks > 0;
    return none_valid && many_masks;
  }

  /*! \brief Check if the capacity is reached

    The capacity is reached when `_num_masks` coincides with the array size.
    Set to nodiscard since this check should be used to determine if it is
    safe to continue updating the SPFD with further coverage.
  */
  [[nodiscard]] inline bool is_saturated() const
  {
    return _num_masks >= capacity;
  }

  /*! \brief Return the number of pairs to be distinguished

    `_num_pairs` is updated by all the methods modifying the SPFD.
  */
  [[nodiscard]] inline uint32_t get_num_pairs() const
  {
    return _num_pairs;
  }

  /*! \brief Count and return the number of pairs to be distinguished
   */
  [[nodiscard]] uint32_t count_pairs()
  {
    _num_pairs = 0;
    PRAGMA_UNROLL
    for ( auto i = 0u; i < _num_masks; ++i )
    {
      _num_pairs += count_pairs( i );
    }
    return _num_pairs;
  }

  /*! \brief Evaluate the number of pairs to be distinguished after removing a given onset
   */
  [[nodiscard]] uint32_t evaluate( TT const& onset_other )
  {
    uint32_t res = 0;
    PRAGMA_UNROLL
    for ( auto i = 0u; i < _num_masks; ++i )
    {
      if ( is_valid( i ) )
      {
        TT const mask_1 = _masks[i] & onset_other;
        TT const mask_0 = _masks[i] & ~onset_other;
        res += kitty::count_ones( _on_set & mask_1 ) * kitty::count_ones( _off_set & mask_1 );
        res += kitty::count_ones( _on_set & mask_0 ) * kitty::count_ones( _off_set & mask_0 );
      }
    }
    return res;
  }

private:
  /* Check if the i-th PFD exists and is not trivial */
  [[nodiscard]] inline bool is_valid( uint32_t i )
  {
    if ( i < _num_masks )
      return _valid[i];
    return false;
  }

  /* Count the number of pairs distinguished by the i-th PFD */
  [[nodiscard]] inline uint32_t count_pairs( uint32_t i )
  {
    if ( !is_valid( i ) )
      return 0;
    uint32_t const num_ones = kitty::count_ones( _on_set & _masks[i] );
    uint32_t const num_zeros = kitty::count_ones( _off_set & _masks[i] );
    return num_ones * num_zeros;
  }

  /* Check if the i-th PFD is saturated */
  [[nodiscard]] inline bool is_constant( uint32_t i )
  {
    bool const is_0 = kitty::is_const0( _masks[i] & _on_set );
    bool const is_1 = kitty::is_const0( _masks[i] & _off_set );
    return is_0 || is_1;
  }

private:
  TT _care_set;
  TT _off_set;
  TT _on_set;
  std::array<TT, capacity> _masks;
  std::array<bool, capacity> _valid;
  uint32_t _num_masks;
  uint32_t _num_pairs;
  uint32_t _num_valid;
};

#undef PRAGMA_UNROLL

} // namespace kitty