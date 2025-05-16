/* kitty: C++ truth table library
 * Copyright (C) 2017-2020  EPFL
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

#include <gtest/gtest.h>

#include <chrono>

#include <kitty/bit_operations.hpp>
#include <kitty/dynamic_truth_table.hpp>
#include <kitty/simd_operations.hpp>
#include <kitty/static_truth_table.hpp>

#include "utility.hpp"

using namespace kitty;

class SIMDTest : public kitty::testing::Test
{
  static constexpr auto num_cases = 100u;

protected:
  template<simd::Operation Op, typename FnSisd, typename FnSimd, typename TT>
  void test_noreturn( FnSisd fn_sisd, FnSimd fn_simd, TT& tt ) const
  {
    TT tt1 = tt.construct();
    TT tt2 = tt.construct();
    double time_diff = 0;
    double time_sisd = 0;
    double time_simd = 0;
    for ( auto i = 0u; i < num_cases; ++i )
    {
      create_random( tt1, i );
      tt2 = tt1;

      run_noreturn_with_time<FnSisd, TT>( fn_sisd, tt1, time_sisd );
      run_noreturn_with_time<FnSimd, TT>( fn_simd, tt2, time_simd );

      time_diff += ( time_simd - time_sisd ) / time_sisd / static_cast<double>( num_cases );
    }
#if KITTY_HAS_AVX2 && !defined( _MSC_VER )
    if ( simd::has_avx2_cached() && simd::use_avx2_cached<Op, TT>( tt, tt.num_vars() ) )
    {
      EXPECT_LE( time_diff, 0 );
    }
#endif
  }

  template<simd::Operation Op, typename FnSisd, typename FnSimd, typename TT>
  void test_unary( FnSisd fn_sisd, FnSimd fn_simd, TT tt ) const
  {
    double time_diff = 0;
    double time_sisd = 0;
    double time_simd = 0;
    TT tt1 = tt.construct();
    for ( auto i = 0u; i < num_cases; ++i )
    {
      create_random( tt1, i );

      auto res_sisd = run_with_time<FnSisd, TT>( fn_sisd, tt1, time_sisd );
      auto res_simd = run_with_time<FnSimd, TT>( fn_simd, tt1, time_simd );
      EXPECT_EQ( res_simd, res_sisd );

      time_diff += ( time_simd - time_sisd ) / time_sisd / static_cast<double>( num_cases );
    }
#if KITTY_HAS_AVX2 && !defined( _MSC_VER )
    if ( simd::has_avx2_cached() && simd::use_avx2_cached<Op, TT>( tt, tt.num_vars() ) )
    {
      EXPECT_LE( time_diff, 0 );
    }
#endif
  }

  template<simd::Operation Op, typename FnSisd, typename FnSimd, typename TT>
  void test_binary( FnSisd fn_sisd, FnSimd fn_simd, TT tt ) const
  {
    double time_diff = 0;
    double time_sisd = 0;
    double time_simd = 0;
    TT tt1 = tt.construct();
    TT tt2 = tt.construct();
    for ( auto i = 0u; i < num_cases; ++i )
    {
      create_random( tt1, i );

      auto res_sisd = run_with_time<FnSisd, TT>( fn_sisd, tt1, tt2, time_sisd );
      auto res_simd = run_with_time<FnSimd, TT>( fn_simd, tt1, tt2, time_simd );
      EXPECT_EQ( res_simd, res_sisd );

      time_diff += ( time_simd - time_sisd ) / time_sisd / static_cast<double>( num_cases );
    }
#if KITTY_HAS_AVX2 && !defined( _MSC_VER )
    if ( simd::has_avx2_cached() && simd::use_avx2_cached<Op, TT>( tt, tt.num_vars() ) && !)
    {
      EXPECT_LE( time_diff, 0 );
      std::cout << "time diff: " << time_diff << std::endl;
    }
#endif
  }

  template<typename F, typename TT>
  auto run_with_time( F func, TT& tt, double& t ) const
  {
    auto start = std::chrono::high_resolution_clock::now();
    auto const res = func( tt );
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    t = elapsed.count();
    return res;
  }

  template<typename F, typename TT>
  auto run_with_time( F func, TT& tt1, TT& tt2, double& t ) const
  {
    auto start = std::chrono::high_resolution_clock::now();
    auto const res = func( tt1, tt2 );
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    t = elapsed.count();
    return res;
  }

  template<typename F, typename TT>
  void run_noreturn_with_time( F func, TT& tt, double& t ) const
  {
    auto start = std::chrono::high_resolution_clock::now();
    func( tt );
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    t = elapsed.count();
  }
};

TEST_F( SIMDTest, simd_set_zero_large )
{

  using TTS = static_truth_table<12u>;
  using TTD = dynamic_truth_table;
  TTS tts;
  simd::test_avx2_advantage( tts, 12u );
  test_noreturn<simd::Operation::CONST0>( [&]( TTS& t )
                                          { t = t ^ t; },
                                          [&]( TTS& t )
                                          { simd::set_zero<TTS>( t ); },
                                          tts );
  TTD ttd( 12u );
  simd::test_avx2_advantage( ttd, 12u );

  test_noreturn<simd::Operation::CONST0>( [&]( TTD& t )
                                          { t = t ^ t; },
                                          [&]( TTD& t )
                                          { simd::set_zero<TTD>( t ); },
                                          ttd );
}

TEST_F( SIMDTest, simd_set_ones_large )
{

  using TTS = static_truth_table<12u>;
  using TTD = dynamic_truth_table;
  TTS tts;
  simd::test_avx2_advantage( tts, 12u );
  test_noreturn<simd::Operation::CONST1>( [&]( TTS& t )
                                          { t = t ^ ~t; },
                                          [&]( TTS& t )
                                          { simd::set_ones<TTS>( t ); },
                                          tts );
  TTD ttd( 12u );
  simd::test_avx2_advantage( ttd, 12u );

  test_noreturn<simd::Operation::CONST1>( [&]( TTD& t )
                                          { t = t ^ ~t; },
                                          [&]( TTD& t )
                                          { simd::set_ones<TTD>( t ); },
                                          ttd );
}

TEST_F( SIMDTest, simd_binary_and_large )
{

  using TTS = static_truth_table<12u>;
  using TTD = dynamic_truth_table;
  TTS tts;
  simd::test_avx2_advantage( tts, 12u );

  test_binary<simd::Operation::AND>( []( const TTS& t1, const TTS& t2 )
                                     { return t1 & t2; },
                                     []( const TTS& t1, const TTS& t2 )
                                     { return simd::binary_and<TTS>( t1, t2 ); },
                                     tts );
  TTD ttd( 12u );
  simd::test_avx2_advantage( ttd, 12u );

  test_binary<simd::Operation::AND>( []( const TTD& t1, const TTD& t2 )
                                     { return t1 & t2; },
                                     []( const TTD& t1, const TTD& t2 )
                                     { return simd::binary_and<TTD>( t1, t2 ); },
                                     ttd );
}

TEST_F( SIMDTest, simd_binary_xor_large )
{
  using TTS = static_truth_table<12u>;
  using TTD = dynamic_truth_table;
  TTS tts;
  simd::test_avx2_advantage( tts, 12u );

  test_binary<simd::Operation::XOR>( []( const TTS& t1, const TTS& t2 )
                                     { return t1 ^ t2; },
                                     []( const TTS& t1, const TTS& t2 )
                                     { return simd::binary_xor<TTS>( t1, t2 ); },
                                     tts );
  TTD ttd( 12u );
  simd::test_avx2_advantage( ttd, 12u );

  test_binary<simd::Operation::XOR>( []( const TTD& t1, const TTD& t2 )
                                     { return t1 ^ t2; },
                                     []( const TTD& t1, const TTD& t2 )
                                     { return simd::binary_xor<TTD>( t1, t2 ); },
                                     ttd );
}

TEST_F( SIMDTest, simd_binary_or_large )
{

  using TTS = static_truth_table<12u>;
  using TTD = dynamic_truth_table;
  TTS tts;
  simd::test_avx2_advantage( tts, 12u );
  test_binary<simd::Operation::OR>( []( const TTS& t1, const TTS& t2 )
                                    { return t1 | t2; },
                                    []( const TTS& t1, const TTS& t2 )
                                    { return simd::binary_or<TTS>( t1, t2 ); },
                                    tts );
  TTD ttd( 12u );
  simd::test_avx2_advantage( ttd, 12u );

  test_binary<simd::Operation::OR>( []( const TTD& t1, const TTD& t2 )
                                    { return t1 | t2; },
                                    []( const TTD& t1, const TTD& t2 )
                                    { return simd::binary_or<TTD>( t1, t2 ); },
                                    ttd );
}

TEST_F( SIMDTest, simd_binary_lt_large )
{
  using TTS = static_truth_table<12u>;
  using TTD = dynamic_truth_table;
  TTS tts;
  simd::test_avx2_advantage( tts, 12u );
  test_binary<simd::Operation::LT>( []( const TTS& t1, const TTS& t2 )
                                    { return ~t1 & t2; },
                                    []( const TTS& t1, const TTS& t2 )
                                    { return simd::binary_lt<TTS>( t1, t2 ); },
                                    tts );
  TTD ttd( 12u );
  simd::test_avx2_advantage( ttd, 12u );

  test_binary<simd::Operation::LT>( []( const TTD& t1, const TTD& t2 )
                                    { return ~t1 & t2; },
                                    []( const TTD& t1, const TTD& t2 )
                                    { return simd::binary_lt<TTD>( t1, t2 ); },
                                    ttd );
}
