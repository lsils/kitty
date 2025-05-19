/* kitty: C++ truth table library
 * Copyright (C) 2017-2025  EPFL
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

#include <algorithm>
#include <iostream>

#include <kitty/esop.hpp>
#include <kitty/print.hpp>
#include <kitty/spectral.hpp>
#include <kitty/static_truth_table.hpp>

#include "utility.hpp"

using namespace kitty;

class SpectralTest : public kitty::testing::Test
{
};

TEST_F( SpectralTest, to_and_from_truth_table )
{
  static_truth_table<5> tt;
  for ( auto i = 0u; i < 1000u; ++i )
  {
    create_random( tt );
    const auto s = detail::spectrum::from_truth_table( tt );
    auto tt2 = tt.construct();
    s.to_truth_table( tt2 );
    EXPECT_EQ( tt, tt2 );
  }
}

TEST_F( SpectralTest, apply_permutation )
{
  static_truth_table<5> tt;
  for ( auto i = 0u; i < 5u; ++i )
  {
    create_random( tt );
    auto s = detail::spectrum::from_truth_table( tt );

    s.permutation( 1 << 2, 1 << 3 );
    swap_inplace( tt, 2, 3 );

    auto tt2 = tt.construct();
    s.to_truth_table( tt2 );
    EXPECT_EQ( tt, tt2 );
  }
}

TEST_F( SpectralTest, apply_input_negation )
{
  static_truth_table<5> tt;
  for ( auto i = 0u; i < 5u; ++i )
  {
    create_random( tt );
    auto s = detail::spectrum::from_truth_table( tt );

    s.input_negation( 1 << 2 );
    flip_inplace( tt, 2 );

    auto tt2 = tt.construct();
    s.to_truth_table( tt2 );
    EXPECT_EQ( tt, tt2 );
  }
}

TEST_F( SpectralTest, apply_output_negation )
{
  static_truth_table<5> tt;
  for ( auto i = 0u; i < 5u; ++i )
  {
    create_random( tt );
    auto s = detail::spectrum::from_truth_table( tt );

    s.output_negation();
    tt = ~tt;

    auto tt2 = tt.construct();
    s.to_truth_table( tt2 );
    EXPECT_EQ( tt, tt2 );
  }
}

TEST_F( SpectralTest, apply_disjoint_translation )
{
  static_truth_table<5> tt;
  for ( auto i = 0u; i < 5u; ++i )
  {
    create_random( tt );
    auto s = detail::spectrum::from_truth_table( tt );

    s.disjoint_translation( 1 << 3 );

    tt ^= nth<5>( 3 );

    auto tt2 = tt.construct();
    s.to_truth_table( tt2 );
    EXPECT_EQ( tt, tt2 );
  }
}

TEST_F( SpectralTest, canonization )
{
  EXPECT_EQ( exact_spectral_canonization( from_hex<3>( "57" ) ), from_hex<3>( "80" ) );
}

TEST_F( SpectralTest, transformations )
{
  static_truth_table<5> tt;
  std::vector<detail::spectral_operation> transforms;

  for ( auto i = 0; i < 100; ++i )
  {
    create_random( tt );
    auto tt_canon = exact_spectral_canonization( tt, [&transforms]( const std::vector<detail::spectral_operation>& best_transforms )
                                                 { transforms = best_transforms; } );

    auto s = detail::spectrum::from_truth_table( tt );
    std::for_each( transforms.begin(), transforms.end(), [&s]( const auto& t )
                   { s.apply( t ); } );

    auto tt2 = tt.construct();
    s.to_truth_table( tt2 );
    EXPECT_EQ( tt_canon, tt2 );

    s = detail::spectrum::from_truth_table( tt2 );
    std::for_each( transforms.rbegin(), transforms.rend(), [&s]( const auto& t )
                   { s.apply( t ); } );

    s.to_truth_table( tt2 );
    EXPECT_EQ( tt, tt2 );
  }
}

TEST_F( SpectralTest, rademacher_walsh_spectrum )
{
  EXPECT_EQ( rademacher_walsh_spectrum( from_hex<3>( "e8" ) ), std::vector<int32_t>( { 0, 4, 4, 0, 4, 0, 0, -4 } ) );
}

TEST_F( SpectralTest, autocorrelation_spectrum )
{
  EXPECT_EQ( autocorrelation_spectrum( from_hex<3>( "6a" ) ), std::vector<int32_t>( { 8, -8, 0, 0, 0, 0, 0, 0 } ) );
}

TEST_F( SpectralTest, distribution_test )
{
  EXPECT_EQ( spectrum_distribution( rademacher_walsh_spectrum( from_hex<5>( "aa55aa55" ) ) ), std::vector<uint32_t>( { 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 } ) );
  EXPECT_EQ( spectrum_distribution( rademacher_walsh_spectrum( from_hex<5>( "aa55ab55" ) ) ), std::vector<uint32_t>( { 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 } ) );
  EXPECT_EQ( spectrum_distribution( rademacher_walsh_spectrum( from_hex<5>( "aa55bb55" ) ) ), std::vector<uint32_t>( { 16, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 } ) );
  EXPECT_EQ( spectrum_distribution( rademacher_walsh_spectrum( from_hex<5>( "aa5dbb55" ) ) ), std::vector<uint32_t>( { 0, 24, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 } ) );
  EXPECT_EQ( spectrum_distribution( rademacher_walsh_spectrum( from_hex<5>( "aaddbb55" ) ) ), std::vector<uint32_t>( { 24, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 } ) );
  EXPECT_EQ( spectrum_distribution( rademacher_walsh_spectrum( from_hex<5>( "aa5dbb51" ) ) ), std::vector<uint32_t>( { 12, 0, 16, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 } ) );
  EXPECT_EQ( spectrum_distribution( rademacher_walsh_spectrum( from_hex<5>( "2a5dbb51" ) ) ), std::vector<uint32_t>( { 0, 20, 0, 10, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 } ) );

  EXPECT_EQ( spectrum_distribution( autocorrelation_spectrum( from_hex<5>( "aa55aa55" ) ) ), std::vector<uint32_t>( { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32 } ) );
  EXPECT_EQ( spectrum_distribution( autocorrelation_spectrum( from_hex<5>( "aa55ab55" ) ) ), std::vector<uint32_t>( { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 0, 1 } ) );
  EXPECT_EQ( spectrum_distribution( autocorrelation_spectrum( from_hex<5>( "aa55bb55" ) ) ), std::vector<uint32_t>( { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 2 } ) );
  EXPECT_EQ( spectrum_distribution( autocorrelation_spectrum( from_hex<5>( "aa5dbb55" ) ) ), std::vector<uint32_t>( { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 3, 0, 1 } ) );
  EXPECT_EQ( spectrum_distribution( autocorrelation_spectrum( from_hex<5>( "aaddbb55" ) ) ), std::vector<uint32_t>( { 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 4 } ) );
  EXPECT_EQ( spectrum_distribution( autocorrelation_spectrum( from_hex<5>( "aa5dbb51" ) ) ), std::vector<uint32_t>( { 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 6, 0, 0, 0, 1 } ) );
  EXPECT_EQ( spectrum_distribution( autocorrelation_spectrum( from_hex<5>( "2a5dbb51" ) ) ), std::vector<uint32_t>( { 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 10, 0, 0, 0, 0, 0, 1 } ) );
}

TEST_F( SpectralTest, spectral_class )
{
  const auto test_one = []( uint32_t num_vars, uint64_t word, uint32_t index )
  {
    dynamic_truth_table func( num_vars );
    create_from_words( func, &word, &word + 1 );
    EXPECT_EQ( get_spectral_class( func ), index );
  };

  for ( auto v = 0u; v <= 5u; ++v )
  {
    const auto& repr = detail::spectral_repr[v];
    for ( auto c = 0u; c < repr.size(); ++c )
    {
      test_one( v, repr[c], c );
    }
  }
}

TEST_F( SpectralTest, spectral_representative )
{
  const auto test_one = []( uint32_t num_vars, uint64_t word )
  {
    dynamic_truth_table func( num_vars );
    create_from_words( func, &word, &word + 1 );
    EXPECT_EQ( spectral_representative( func ), func );
  };

  for ( auto v = 0u; v <= 5u; ++v )
  {
    const auto& repr = detail::spectral_repr[v];
    for ( auto c = 0u; c < repr.size(); ++c )
    {
      test_one( v, repr[c] );
    }
  }
}

template<uint32_t NumVars>
static void algebraic_normal_form_test()
{
  static_truth_table<NumVars> tt;
  for ( auto i = 0u; i < 100u; ++i )
  {
    create_random( tt );
    const auto anf = detail::algebraic_normal_form( tt );
    const auto tt2 = detail::algebraic_normal_form( anf );
    EXPECT_EQ( tt, tt2 );
  }
}

TEST_F( SpectralTest, algebraic_normal_form )
{
  algebraic_normal_form_test<0>();
  algebraic_normal_form_test<1>();
  algebraic_normal_form_test<2>();
  algebraic_normal_form_test<3>();
  algebraic_normal_form_test<4>();
  algebraic_normal_form_test<5>();
  algebraic_normal_form_test<6>();
  algebraic_normal_form_test<7>();
}

template<uint32_t NumVars>
void hybrid_exact_spectral_canonization()
{
  const auto& repr = detail::spectral_repr[NumVars];
  for ( auto word : repr )
  {
    static_truth_table<NumVars> tt1;
    create_from_words( tt1, &word, &word + 1 );
    const auto tt2 = exact_spectral_canonization( tt1 );
    const auto tt3 = hybrid_exact_spectral_canonization( tt1 );
    const auto tt4 = hybrid_exact_spectral_canonization( tt2 );
    EXPECT_EQ( tt3, tt4 );
  }
}

TEST_F( SpectralTest, hybrid_exact_spectral_canonization )
{
  hybrid_exact_spectral_canonization<3>();
  hybrid_exact_spectral_canonization<4>();
  // hybrid_exact_spectral_canonization<5>();
}

TEST_F( SpectralTest, hybrid_exact_spectral_canonization_with_transformation )
{
  static_truth_table<5> tt;
  std::vector<detail::spectral_operation> transforms;

  for ( auto i = 0; i < 100; ++i )
  {
    create_random( tt );
    const auto r = hybrid_exact_spectral_canonization( tt, [&]( const auto& _transforms )
                                                       { transforms = _transforms; } );

    auto s = detail::spectrum::from_truth_table( tt );
    std::for_each( transforms.begin(), transforms.end(), [&s]( const auto& t )
                   { s.apply( t ); } );
    auto tt2 = tt.construct();
    s.to_truth_table( tt2 );
    EXPECT_EQ( r, tt2 );

    s = detail::spectrum::from_truth_table( tt2 );
    std::for_each( transforms.rbegin(), transforms.rend(), [&s]( const auto& t )
                   { s.apply( t ); } );

    s.to_truth_table( tt2 );
    EXPECT_EQ( tt, tt2 );
  }
}
