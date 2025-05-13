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
  \file simd_operations.hpp
  \brief Implements efficient alternatives of some common operations using SIMD

  \author Andrea Costamagna
*/

#pragma once

#include <cstdint>
#include <cassert>
#include <numeric>
#include <algorithm>

#include "static_truth_table.hpp"
#include "partial_truth_table.hpp"
#include "dynamic_truth_table.hpp"
#include "detail/mscfix.hpp"

/*! Check if the code is compiled on a platform that might support AVX2 */
#ifndef KITTY_HAS_AVX2
#if defined( __x86_64__ ) || defined( _M_X64 ) || defined( _MSC_VER )
#define KITTY_HAS_AVX2 1
#else
#define KITTY_HAS_AVX2 0
#endif
#endif

#if KITTY_HAS_AVX2
#if defined( _MSC_VER )
#include <intrin.h>
#else
#include <immintrin.h>
#include <cpuid.h>
#endif
#endif

namespace kitty
{

namespace simd
{
/*! Check if AVX2 is supported on this machine. */
inline bool has_avx2_cached()
{
#if KITTY_HAS_AVX2
#if defined( _MSC_VER )
  static const bool cached = []
  {
    int cpuInfo[4];
    __cpuid( cpuInfo, 0 );
    if ( cpuInfo[0] < 7 )
      return false;
    __cpuidex( cpuInfo, 7, 0 );
    return ( cpuInfo[1] & ( 1 << 5 ) ) != 0;
  }();
  return cached;
#else
  static const bool cached = []
  {
    unsigned int eax, ebx, ecx, edx;
    unsigned int max_leaf = __get_cpuid_max( 0, nullptr );
    if ( !max_leaf || max_leaf < 7 )
      return false;
    __cpuid_count( 7, 0, eax, ebx, ecx, edx );
    return ( ebx & ( 1 << 5 ) ) != 0;
  }();
  return cached;
#endif
#else
  return false;
#endif
}

/*! Enumeration type to minimize code redundancies. */
enum class BinaryOp
{
  AND,
  OR,
  XOR,
  LT
};

/*! Compile-time dispatch of the binary operations. */
#if KITTY_HAS_AVX2
template<BinaryOp binary_op>
inline __m256i get_simd_op( __m256i vr, __m256i v2 )
{
  if constexpr ( binary_op == BinaryOp::AND )
    return _mm256_and_si256( vr, v2 );
  else if constexpr ( binary_op == BinaryOp::OR )
    return _mm256_or_si256( vr, v2 );
  else if constexpr ( binary_op == BinaryOp::XOR )
    return _mm256_xor_si256( vr, v2 );
  else if constexpr ( binary_op == BinaryOp::LT )
    return _mm256_andnot_si256( vr, v2 );
  else
    static_assert( binary_op != binary_op, "Unsupported BinaryOp" );
}
#endif

/*! Perform the vectorized version of the binary operations using AVX2. */
template<BinaryOp binary_op, typename TT, typename ScalarOp>
inline TT bitwise_binop( const TT& tta, const TT& ttb, ScalarOp&& scalar_op )
{
  TT result = tta;
  size_t size = tta.num_blocks();
  auto& datar = result._bits;
  const auto& data2 = ttb._bits;

  size_t i = 0;
#if KITTY_HAS_AVX2
  if ( has_avx2_cached() && size >= 4 )
  {
    for ( ; i + 4 <= size; i += 4 )
    {
      __m256i vr = _mm256_loadu_si256( reinterpret_cast<const __m256i*>( &datar[i] ) );
      __m256i v2 = _mm256_loadu_si256( reinterpret_cast<const __m256i*>( &data2[i] ) );
      vr = get_simd_op<binary_op>( vr, v2 );
      _mm256_storeu_si256( reinterpret_cast<__m256i*>( &datar[i] ), vr );
    }
  }
#endif
  /* Fallback to the scalar version for the remaining elements. */
  for ( ; i < size; ++i )
  {
    datar[i] = scalar_op( datar[i], data2[i] );
  }

  result.mask_bits();
  return result;
}

/*! \brief Perform a vectorized bitwise AND between two truth tables.
 *
 * Computes the bitwise AND \f$tt_a \land tt_b\f$ using 256-bit AVX2 registers.
 * Each register processes four 64-bit words in parallel, enabling efficient
 * parallel computation across the truth tables.
 *
 * \param tta First truth table.
 * \param ttb Second truth table.
 */
template<typename TT>
inline TT bitwise_and( const TT& tta, const TT& ttb )
{
  return bitwise_binop<BinaryOp::AND>(
      tta, ttb,
      []( uint64_t a, uint64_t b )
      { return a & b; } );
}

/*! \brief Perform a vectorized bitwise OR between two truth tables.
 *
 * Computes the bitwise OR \f$tt_a \lor tt_b\f$ using 256-bit AVX2 registers.
 * Each register processes four 64-bit words in parallel, enabling efficient
 * parallel computation across the truth tables.
 *
 * \param tta First truth table.
 * \param ttb Second truth table.
 */
template<typename TT>
inline TT bitwise_or( const TT& tta, const TT& ttb )
{
  return bitwise_binop<BinaryOp::OR>(
      tta, ttb,
      []( uint64_t a, uint64_t b )
      { return a | b; } );
}

/*! \brief Perform a vectorized bitwise XOR between two truth tables.
 *
 * Computes the bitwise XOR \f$tt_a \lxor tt_b\f$ using 256-bit AVX2 registers.
 * Each register processes four 64-bit words in parallel, enabling efficient
 * parallel computation across the truth tables.
 *
 * \param tta First truth table.
 * \param ttb Second truth table.
 */
template<typename TT>
inline TT bitwise_xor( const TT& tta, const TT& ttb )
{
  return bitwise_binop<BinaryOp::XOR>(
      tta, ttb,
      []( uint64_t a, uint64_t b )
      { return a ^ b; } );
}

/*! \brief Perform a vectorized bitwise LT ( Lower Than ) between two truth tables.
 *
 * Computes the bitwise LT \f$ ~tt_a \land tt_b\f$ using 256-bit AVX2 registers.
 * Each register processes four 64-bit words in parallel, enabling efficient
 * parallel computation across the truth tables.
 *
 * \param tta First truth table.
 * \param ttb Second truth table.
 */
template<typename TT>
inline TT bitwise_lt( const TT& tta, const TT& ttb )
{
  return bitwise_binop<BinaryOp::LT>(
      tta, ttb,
      []( uint64_t a, uint64_t b )
      { return ~a & b; } );
}

/*! \brief Perform a vectorized inversion of a truth tables.
 *
 * Computes the inverse \f$ ~tt \f$ using 256-bit AVX2 registers.
 * Each register processes four 64-bit words in parallel, enabling efficient
 * parallel inversion across the truth tables.
 *
 * \param tt Truth table.
 */
template<typename TT>
inline TT unary_not( const TT& tt )
{
  TT result = tt;
  size_t size = tt.num_blocks();
  auto& data = result._bits;

  size_t i = 0;
#if KITTY_HAS_AVX2
  if ( has_avx2_cached() )
  {
    const __m256i all_ones = _mm256_set1_epi64x( -1 );
    for ( ; i + 4 <= size; i += 4 )
    {
      __m256i v = _mm256_loadu_si256( reinterpret_cast<const __m256i*>( &data[i] ) );
      v = _mm256_xor_si256( v, all_ones );
      _mm256_storeu_si256( reinterpret_cast<__m256i*>( &data[i] ), v );
    }
  }
#endif

  for ( ; i < size; ++i )
  {
    data[i] = ~data[i];
  }
  result.mask_bits();
  return result;
}

/*! Vectorized set of a truth-table to a constant value. */
template<typename TT, uint32_t VAL>
inline void set_const( TT& tt )
{
  size_t size = tt.num_blocks();
  auto& data = tt._bits;

  size_t i = 0;
#if KITTY_HAS_AVX2
  if ( has_avx2_cached() )
  {
    __m256i v;
    if constexpr ( VAL == 0 )
    {
      v = _mm256_setzero_si256(); // Set the vector to zero
    }
    else
    {
      v = _mm256_set1_epi64x( -1 );
    }

    for ( ; i + 4 <= size; i += 4 )
    {
      _mm256_storeu_si256( reinterpret_cast<__m256i*>( &data[i] ), v );
    }
  }
#endif
  uint64_t v;

  if constexpr ( VAL == 0 )
  {
    v = 0;
  }
  else
  {
    v = (uint64_t)( -1 );
  }

  // Handle remaining elements
  for ( ; i < size; ++i )
  {
    data[i] = v;
  }
}

/*! \brief Reset all the bits of a truth table to 0 through vectorization.
 *
 * Set all the bits of a truth table to 0 using 256-bit AVX2 registers.
 * Each register processes four 64-bit words in parallel, enabling efficiently
 * setting the truth table to the desird value.
 *
 * \param tt Truth table.
 */
template<typename TT>
inline void set_zero( TT& tt )
{
  set_const<TT, 0>( tt );
}

/*! \brief Reset all the bits of a truth table to 1 through vectorization.
 *
 * Set all the bits of a truth table to 1 using 256-bit AVX2 registers.
 * Each register processes four 64-bit words in parallel, enabling efficiently
 * setting the truth table to the desird value.
 *
 * \param tt Truth table.
 */
template<typename TT>
inline void set_ones( TT& tt )
{
  set_const<TT, 1>( tt );
}

/*! Implementation of unary NOT for small static truth tables for compatibility. */
template<uint32_t NumVars>
inline kitty::static_truth_table<NumVars> unary_not( kitty::static_truth_table<NumVars, true> const& tt )
{
  return ~tt;
}

/*! Implementation of bitwise AND for small static truth tables for compatibility. */
template<uint32_t NumVars>
inline kitty::static_truth_table<NumVars> bitwise_and( const kitty::static_truth_table<NumVars, true>& tta, const kitty::static_truth_table<NumVars, true>& ttb )
{
  return tta & ttb;
}

/*! Implementation of bitwise OR for small static truth tables for compatibility. */
template<uint32_t NumVars>
inline kitty::static_truth_table<NumVars> bitwise_or( const kitty::static_truth_table<NumVars, true>& tta, const kitty::static_truth_table<NumVars, true>& ttb )
{
  return tta | ttb;
}

/*! Implementation of bitwise XOR for small static truth tables for compatibility. */
template<uint32_t NumVars>
inline kitty::static_truth_table<NumVars> bitwise_xor( const kitty::static_truth_table<NumVars, true>& tta, const kitty::static_truth_table<NumVars, true>& ttb )
{
  return tta ^ ttb;
}

/*! Implementation of bitwise LT for small static truth tables for compatibility. */
template<uint32_t NumVars>
inline kitty::static_truth_table<NumVars> bitwise_lt( const kitty::static_truth_table<NumVars, true>& tta, const kitty::static_truth_table<NumVars, true>& ttb )
{
  return ~tta & ttb;
}

/*! Implementation set to constant 1 for small static truth tables. */
template<uint32_t NumVars>
inline void set_ones( kitty::static_truth_table<NumVars, true>& tt )
{
  tt |= ~tt;
}

/*! Implementation set to constant 0 for small static truth tables. */
template<uint32_t NumVars>
inline void set_zero( kitty::static_truth_table<NumVars, true>& tt )
{
  tt ^= tt;
}

} // namespace simd

} // namespace kitty
