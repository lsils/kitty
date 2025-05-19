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

#include <kitty/constructors.hpp>
#include <kitty/dynamic_truth_table.hpp>
#include <kitty/decomposition.hpp>
#include <kitty/print.hpp>
#include <kitty/static_truth_table.hpp>

#include "utility.hpp"

using namespace kitty;

class DecompositionTest : public kitty::testing::Test
{
};

TEST_F( DecompositionTest, top_down )
{
  EXPECT_EQ( is_top_decomposable( from_hex<3>( "e8" ), 0u ), top_decomposition::none );
  EXPECT_EQ( is_top_decomposable( from_hex<3>( "e8" ), 1u ), top_decomposition::none );
  EXPECT_EQ( is_top_decomposable( from_hex<3>( "e8" ), 2u ), top_decomposition::none );

  kitty::static_truth_table<3> expr1;
  kitty::static_truth_table<3> expr2, expr2_der;
  kitty::static_truth_table<3> expr3, expr3_der;

  create_from_expression( expr1, "(a{bc})" );
  create_from_expression( expr2, "{bc}" );
  create_from_expression( expr3, "c" );

  EXPECT_EQ( is_top_decomposable( expr1, 0u, &expr2_der ), top_decomposition::and_ );
  EXPECT_EQ( expr2, expr2_der );
  EXPECT_EQ( is_top_decomposable( expr2, 1u, &expr3_der ), top_decomposition::or_ );
  EXPECT_EQ( expr3, expr3_der );

  create_from_expression( expr1, "(!a{!bc})" );
  create_from_expression( expr2, "{!bc}" );
  create_from_expression( expr3, "c" );

  EXPECT_EQ( is_top_decomposable( expr1, 0u, &expr2_der ), top_decomposition::lt_ );
  EXPECT_EQ( expr2, expr2_der );
  EXPECT_EQ( is_top_decomposable( expr2, 1u, &expr3_der ), top_decomposition::le_ );
  EXPECT_EQ( expr3, expr3_der );

  create_from_expression( expr1, "[a{bc}]" );
  create_from_expression( expr2, "{bc}" );
  create_from_expression( expr3, "c" );

  EXPECT_EQ( is_top_decomposable( expr1, 0u, &expr2_der ), top_decomposition::xor_ );
  EXPECT_EQ( expr2, expr2_der );
  EXPECT_EQ( is_top_decomposable( expr2, 1u, &expr3_der ), top_decomposition::or_ );
  EXPECT_EQ( expr3, expr3_der );

  auto res = is_top_decomposable<kitty::static_truth_table<3>>( expr1, 0u, nullptr, false );
  EXPECT_EQ( res, top_decomposition::none );
}

TEST_F( DecompositionTest, bottom_up )
{
  kitty::static_truth_table<6> expr1;
  kitty::static_truth_table<6> expr2, expr2_der;
  kitty::static_truth_table<6> expr3, expr3_der;
  kitty::static_truth_table<6> expr4, expr4_der;
  kitty::static_truth_table<6> expr5, expr5_der;
  kitty::static_truth_table<6> expr6, expr6_der;

  create_from_expression( expr1, "[a(b{!c(!d{ef})})]" );
  create_from_expression( expr2, "[a(b{!c(!de)})]" );
  create_from_expression( expr3, "[a(b{!cd})]" );
  create_from_expression( expr4, "[a(bc)]" );
  create_from_expression( expr5, "[ab]" );
  create_from_expression( expr6, "a" );

  EXPECT_EQ( is_bottom_decomposable( expr1, 4u, 5u, &expr2_der ), bottom_decomposition::or_ );
  EXPECT_EQ( expr2, expr2_der );
  EXPECT_EQ( is_bottom_decomposable( expr2, 3u, 4u, &expr3_der ), bottom_decomposition::lt_ );
  EXPECT_EQ( expr3, expr3_der );
  EXPECT_EQ( is_bottom_decomposable( expr3, 2u, 3u, &expr4_der ), bottom_decomposition::le_ );
  EXPECT_EQ( expr4, expr4_der );
  EXPECT_EQ( is_bottom_decomposable( expr4, 1u, 2u, &expr5_der ), bottom_decomposition::and_ );
  EXPECT_EQ( expr5, expr5_der );
  EXPECT_EQ( is_bottom_decomposable( expr5, 0u, 1u, &expr6_der ), bottom_decomposition::xor_ );
  EXPECT_EQ( expr6, expr6_der );

  auto res = is_bottom_decomposable<kitty::static_truth_table<6>>( expr5, 0u, 1u, nullptr, false );
  EXPECT_EQ( res, bottom_decomposition::none );
}

TEST_F( DecompositionTest, combined )
{
  kitty::static_truth_table<6> expr1;
  kitty::static_truth_table<6> expr2, expr2_der;
  kitty::static_truth_table<6> expr3, expr3_der;
  kitty::static_truth_table<6> expr4, expr4_der;
  kitty::static_truth_table<6> expr5, expr5_der;
  kitty::static_truth_table<6> expr6, expr6_der;

  create_from_expression( expr1, "{[a{bc}]{d(ef)}}" );
  create_from_expression( expr2, "{[a{bc}]{de}}" );
  create_from_expression( expr3, "{[ab]{de}}" );
  create_from_expression( expr4, "{[ab]d}" );
  create_from_expression( expr5, "[ab]" );
  create_from_expression( expr6, "a" );

  EXPECT_EQ( is_bottom_decomposable( expr1, 4u, 5u, &expr2_der ), bottom_decomposition::and_ );
  EXPECT_EQ( expr2, expr2_der );
  EXPECT_EQ( is_bottom_decomposable( expr2, 1u, 2u, &expr3_der ), bottom_decomposition::or_ );
  EXPECT_EQ( expr3, expr3_der );
  EXPECT_EQ( is_bottom_decomposable( expr3, 3u, 4u, &expr4_der ), bottom_decomposition::or_ );
  EXPECT_EQ( expr4, expr4_der );
  EXPECT_EQ( is_top_decomposable( expr4, 3u, &expr5_der ), top_decomposition::or_ );
  EXPECT_EQ( expr5, expr5_der );
  EXPECT_EQ( is_top_decomposable( expr5, 1u, &expr6_der ), top_decomposition::xor_ );
  EXPECT_EQ( expr6, expr6_der );
}

TEST_F( DecompositionTest, bi_decomposition )
{
  kitty::static_truth_table<5> tt;
  kitty::static_truth_table<5> dc;

  for ( auto i = 0; i < 50; i++ )
  {
    create_random( tt );
    create_random( dc, i );
    auto f = is_bi_decomposable( tt, dc );

    if ( std::get<1>( f ) == bi_decomposition::none )
    {
      EXPECT_EQ( binary_and( std::get<0>( f ), dc ), binary_and( tt, dc ) );
      continue;
    }

    auto q_and_r = std::get<2>( f );
    auto a = q_and_r[0];
    auto b = q_and_r[2];

    if ( ( std::get<1>( f ) == bi_decomposition::and_ ) || ( std::get<1>( f ) == bi_decomposition::weak_and_ ) )
    {
      EXPECT_EQ( binary_and( std::get<0>( f ), dc ), binary_and( binary_and( a, b ), dc ) );
    }
    else if ( ( std::get<1>( f ) == bi_decomposition::or_ ) || ( std::get<1>( f ) == bi_decomposition::weak_or_ ) )
    {
      EXPECT_EQ( binary_and( std::get<0>( f ), dc ), binary_and( binary_or( a, b ), dc ) );
    }
    else if ( std::get<1>( f ) == bi_decomposition::xor_ )
    {
      EXPECT_EQ( binary_and( std::get<0>( f ), dc ), binary_and( binary_xor( a, b ), dc ) );
    }

    EXPECT_EQ( binary_and( std::get<0>( f ), dc ), binary_and( tt, dc ) );
  }
}

TEST_F( DecompositionTest, bi_decomposition_mc )
{
  kitty::static_truth_table<5> tt;
  kitty::static_truth_table<5> dc;

  for ( auto i = 0; i < 50; i++ )
  {
    create_random( tt );
    create_random( dc, i );
    auto f = is_bi_decomposable_mc( tt, dc );

    if ( std::get<1>( f ) == bi_decomposition::none )
    {
      EXPECT_EQ( binary_and( std::get<0>( f ), dc ), binary_and( tt, dc ) );
      continue;
    }

    auto q_and_r = std::get<2>( f );
    auto a = q_and_r[0];
    auto b = q_and_r[2];

    if ( ( std::get<1>( f ) == bi_decomposition::and_ ) || ( std::get<1>( f ) == bi_decomposition::weak_and_ ) )
    {
      EXPECT_EQ( binary_and( std::get<0>( f ), dc ), binary_and( binary_and( a, b ), dc ) );
    }
    else if ( ( std::get<1>( f ) == bi_decomposition::or_ ) || ( std::get<1>( f ) == bi_decomposition::weak_or_ ) )
    {
      EXPECT_EQ( binary_and( std::get<0>( f ), dc ), binary_and( binary_or( a, b ), dc ) );
    }
    else if ( std::get<1>( f ) == bi_decomposition::xor_ )
    {
      EXPECT_EQ( binary_and( std::get<0>( f ), dc ), binary_and( binary_xor( a, b ), dc ) );
    }

    EXPECT_EQ( binary_and( std::get<0>( f ), dc ), binary_and( tt, dc ) );
  }
}

TEST_F( DecompositionTest, ashenhurst )
{
  using TTg = static_truth_table<3>;
  using TTf = static_truth_table<5>;

  TTf tt, x1, x2, x3, x4, x5;
  create_nth_var( x1, 0 );
  create_nth_var( x2, 1 );
  create_nth_var( x3, 2 );
  create_nth_var( x4, 3 );
  create_nth_var( x5, 4 );

  tt = x1 | x2 | ( x3 & x4 & x5 );
  std::vector<uint32_t> ys_idx{ 2, 3, 4 };

  std::vector<std::pair<TTg, TTg>> decomposition;
  const auto count = kitty::ashenhurst_decomposition( tt, ys_idx, decomposition );
  EXPECT_EQ( 2u, count );
  EXPECT_EQ( 2u, decomposition.size() );

  EXPECT_EQ( decomposition[0].first, from_hex<3>( "ef" ) );
  EXPECT_EQ( decomposition[0].second, from_hex<3>( "7f" ) );
  EXPECT_EQ( decomposition[1].first, from_hex<3>( "fe" ) );
  EXPECT_EQ( decomposition[1].second, from_hex<3>( "80" ) );
}
