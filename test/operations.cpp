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

#include <algorithm>
#include <random>

#include <kitty/dynamic_truth_table.hpp>
#include <kitty/ternary_truth_table.hpp>
#include <kitty/operations.hpp>
#include <kitty/static_truth_table.hpp>
#include <kitty/print.hpp>

#include "utility.hpp"

using namespace kitty;

class OperationsTest : public kitty::testing::Test
{
};

TEST_F( OperationsTest, unary_for_small )
{
  EXPECT_EQ( unary_not( from_hex<2>( "8" ) ), from_hex<2>( "7" ) );
  EXPECT_EQ( unary_not( from_hex<2>( "6" ) ), from_hex<2>( "9" ) );

  EXPECT_EQ( unary_not_if( from_hex<2>( "8" ), true ), from_hex<2>( "7" ) );
  EXPECT_EQ( unary_not_if( from_hex<2>( "6" ), true ), from_hex<2>( "9" ) );

  EXPECT_EQ( unary_not_if( from_hex<2>( "8" ), false ), from_hex<2>( "8" ) );
  EXPECT_EQ( unary_not_if( from_hex<2>( "6" ), false ), from_hex<2>( "6" ) );
}

TEST_F( OperationsTest, binary_for_small )
{
  EXPECT_EQ( binary_and( nth<2>( 0 ), nth<2>( 1 ) ), from_hex<2>( "8" ) );
  EXPECT_EQ( binary_or( nth<2>( 0 ), nth<2>( 1 ) ), from_hex<2>( "e" ) );
  EXPECT_EQ( unary_not( binary_and( nth<2>( 0 ), nth<2>( 1 ) ) ), from_hex<2>( "7" ) );
  EXPECT_EQ( unary_not( binary_or( nth<2>( 0 ), nth<2>( 1 ) ) ), from_hex<2>( "1" ) );

  EXPECT_EQ( binary_and( nth( 2, 0 ), nth( 2, 1 ) ), from_hex( 2, "8" ) );
  EXPECT_EQ( binary_or( nth( 2, 0 ), nth( 2, 1 ) ), from_hex( 2, "e" ) );
  EXPECT_EQ( unary_not( binary_and( nth( 2, 0 ), nth( 2, 1 ) ) ), from_hex( 2, "7" ) );
  EXPECT_EQ( unary_not( binary_or( nth( 2, 0 ), nth( 2, 1 ) ) ), from_hex( 2, "1" ) );
}

TEST_F( OperationsTest, ternary_for_small )
{
  {
    const auto t1 = nth<3>( 0 ), t2 = nth<3>( 1 ), t3 = nth<3>( 2 );

    const auto maj_expr = binary_or( binary_or( binary_and( t1, t2 ), binary_and( t1, t3 ) ), binary_and( t2, t3 ) );
    const auto maj_direct = ternary_majority( t1, t2, t3 );

    EXPECT_EQ( maj_expr, from_hex<3>( "e8" ) );
    EXPECT_EQ( maj_expr, maj_direct );

    const auto ite_expr = binary_or( binary_and( t1, t2 ), binary_and( unary_not( t1 ), t3 ) );
    const auto ite_direct = ternary_ite( t1, t2, t3 );

    EXPECT_EQ( ite_expr, from_hex<3>( "d8" ) );
    EXPECT_EQ( ite_expr, ite_direct );
  }

  {
    const auto t1 = nth( 3, 0 ), t2 = nth( 3, 1 ), t3 = nth( 3, 2 );

    const auto maj_expr = binary_or( binary_or( binary_and( t1, t2 ), binary_and( t1, t3 ) ), binary_and( t2, t3 ) );
    const auto maj_direct = ternary_majority( t1, t2, t3 );

    EXPECT_EQ( maj_expr, from_hex( 3, "e8" ) );
    EXPECT_EQ( maj_expr, maj_direct );

    const auto ite_expr = binary_or( binary_and( t1, t2 ), binary_and( unary_not( t1 ), t3 ) );
    const auto ite_direct = ternary_ite( t1, t2, t3 );

    EXPECT_EQ( ite_expr, from_hex( 3, "d8" ) );
    EXPECT_EQ( ite_expr, ite_direct );
  }
}

TEST_F( OperationsTest, unary_for_large )
{
  EXPECT_EQ( unary_not( from_hex<7>( "80000000000000000000000000000000" ) ), from_hex<7>( "7fffffffffffffffffffffffffffffff" ) );
  EXPECT_EQ( unary_not( from_hex<7>( "66666666666666666666666666666666" ) ), from_hex<7>( "99999999999999999999999999999999" ) );

  EXPECT_EQ( unary_not_if( from_hex<7>( "80000000000000000000000000000000" ), true ), from_hex<7>( "7fffffffffffffffffffffffffffffff" ) );
  EXPECT_EQ( unary_not_if( from_hex<7>( "66666666666666666666666666666666" ), true ), from_hex<7>( "99999999999999999999999999999999" ) );

  EXPECT_EQ( unary_not_if( from_hex<7>( "80000000000000000000000000000000" ), false ), from_hex<7>( "80000000000000000000000000000000" ) );
  EXPECT_EQ( unary_not_if( from_hex<7>( "66666666666666666666666666666666" ), false ), from_hex<7>( "66666666666666666666666666666666" ) );
}

TEST_F( OperationsTest, binary_for_large )
{
  {
    auto tt = nth<7>( 0 );
    for ( auto i = 1; i < 7; ++i )
    {
      tt = binary_and( tt, nth<7>( i ) );
    }
    EXPECT_EQ( tt, from_hex<7>( "80000000000000000000000000000000" ) );
  }

  {
    auto tt = nth( 7, 0 );
    for ( auto i = 1; i < 7; ++i )
    {
      tt = binary_and( tt, nth( 7, i ) );
    }
    EXPECT_EQ( tt, from_hex( 7, "80000000000000000000000000000000" ) );
  }
}

TEST_F( OperationsTest, ternary_for_large )
{
  {
    const auto t1 = nth<7>( 0 ), t2 = nth<7>( 1 ), t3 = nth<7>( 2 );

    const auto maj_expr = binary_or( binary_or( binary_and( t1, t2 ), binary_and( t1, t3 ) ), binary_and( t2, t3 ) );
    const auto maj_direct = ternary_majority( t1, t2, t3 );

    EXPECT_EQ( maj_expr, maj_direct );

    const auto ite_expr = binary_or( binary_and( t1, t2 ), binary_and( unary_not( t1 ), t3 ) );
    const auto ite_direct = ternary_ite( t1, t2, t3 );

    EXPECT_EQ( ite_expr, ite_direct );
  }

  {
    const auto t1 = nth( 7, 0 ), t2 = nth( 7, 1 ), t3 = nth( 7, 2 );

    const auto maj_expr = binary_or( binary_or( binary_and( t1, t2 ), binary_and( t1, t3 ) ), binary_and( t2, t3 ) );
    const auto maj_direct = ternary_majority( t1, t2, t3 );

    EXPECT_EQ( maj_expr, maj_direct );

    const auto ite_expr = binary_or( binary_and( t1, t2 ), binary_and( unary_not( t1 ), t3 ) );
    const auto ite_direct = ternary_ite( t1, t2, t3 );

    EXPECT_EQ( ite_expr, ite_direct );
  }
}

TEST_F( OperationsTest, comparisons )
{
  EXPECT_TRUE( equal( from_hex<3>( "e8" ), from_hex<3>( "e8" ) ) );
  EXPECT_TRUE( equal( from_hex( 3, "e8" ), from_hex( 3, "e8" ) ) );

  EXPECT_TRUE( less_than( from_hex<3>( "e5" ), from_hex<3>( "f6" ) ) );
  EXPECT_TRUE( less_than( from_hex<3>( "e5" ), from_hex<3>( "f5" ) ) );
  EXPECT_TRUE( less_than( from_hex<3>( "e5" ), from_hex<3>( "f4" ) ) );
  EXPECT_TRUE( less_than( from_hex<3>( "e5" ), from_hex<3>( "e6" ) ) );
  EXPECT_FALSE( less_than( from_hex<3>( "e5" ), from_hex<3>( "e5" ) ) );
  EXPECT_FALSE( less_than( from_hex<3>( "e5" ), from_hex<3>( "e4" ) ) );
  EXPECT_FALSE( less_than( from_hex<3>( "e5" ), from_hex<3>( "d6" ) ) );
  EXPECT_FALSE( less_than( from_hex<3>( "e5" ), from_hex<3>( "d5" ) ) );
  EXPECT_FALSE( less_than( from_hex<3>( "e5" ), from_hex<3>( "d4" ) ) );

  EXPECT_TRUE( less_than( from_hex( 3, "e5" ), from_hex( 3, "f6" ) ) );
  EXPECT_TRUE( less_than( from_hex( 3, "e5" ), from_hex( 3, "f5" ) ) );
  EXPECT_TRUE( less_than( from_hex( 3, "e5" ), from_hex( 3, "f4" ) ) );
  EXPECT_TRUE( less_than( from_hex( 3, "e5" ), from_hex( 3, "e6" ) ) );
  EXPECT_FALSE( less_than( from_hex( 3, "e5" ), from_hex( 3, "e5" ) ) );
  EXPECT_FALSE( less_than( from_hex( 3, "e5" ), from_hex( 3, "e4" ) ) );
  EXPECT_FALSE( less_than( from_hex( 3, "e5" ), from_hex( 3, "d6" ) ) );
  EXPECT_FALSE( less_than( from_hex( 3, "e5" ), from_hex( 3, "d5" ) ) );
  EXPECT_FALSE( less_than( from_hex( 3, "e5" ), from_hex( 3, "d4" ) ) );

  EXPECT_TRUE( equal( from_hex<7>( "e92c774439c72c8955906ef92ecefec9" ), from_hex<7>( "e92c774439c72c8955906ef92ecefec9" ) ) );
  EXPECT_TRUE( equal( from_hex( 7, "e92c774439c72c8955906ef92ecefec9" ), from_hex( 7, "e92c774439c72c8955906ef92ecefec9" ) ) );

  EXPECT_TRUE( less_than( from_hex<7>( "e92c774439c72c8955906ef92ecefec9" ), from_hex<7>( "e92c774439c72c8955906ef92edefec9" ) ) );
  EXPECT_FALSE( less_than( from_hex<7>( "e92c774439c72c8955906ef92ecefec9" ), from_hex<7>( "e92c774439c72c8955906ef92ebefec9" ) ) );

  EXPECT_TRUE( less_than( from_hex( 7, "e92c774439c72c8955906ef92ecefec9" ), from_hex( 7, "e92c774439c72c8955906ef92edefec9" ) ) );
  EXPECT_FALSE( less_than( from_hex( 7, "e92c774439c72c8955906ef92ecefec9" ), from_hex( 7, "e92c774439c72c8955906ef92ebefec9" ) ) );
}

TEST_F( OperationsTest, support )
{
  EXPECT_TRUE( is_const0( from_hex<0>( "0" ) ) );
  EXPECT_TRUE( is_const0( from_hex<1>( "0" ) ) );
  EXPECT_TRUE( is_const0( from_hex<2>( "0" ) ) );
  EXPECT_TRUE( is_const0( from_hex<3>( "00" ) ) );
  EXPECT_TRUE( is_const0( from_hex<4>( "0000" ) ) );
  EXPECT_TRUE( is_const0( from_hex<5>( "00000000" ) ) );
  EXPECT_TRUE( is_const0( from_hex<6>( "0000000000000000" ) ) );
  EXPECT_TRUE( is_const0( from_hex<7>( "00000000000000000000000000000000" ) ) );
  EXPECT_TRUE( is_const0( from_hex<8>( "0000000000000000000000000000000000000000000000000000000000000000" ) ) );

  EXPECT_TRUE( has_var( from_hex<3>( "77" ), 0 ) );
  EXPECT_TRUE( has_var( from_hex<3>( "77" ), 1 ) );
  EXPECT_TRUE( !has_var( from_hex<3>( "77" ), 2 ) );

  EXPECT_TRUE( has_var( from_hex( 3, "77" ), 0 ) );
  EXPECT_TRUE( has_var( from_hex( 3, "77" ), 1 ) );
  EXPECT_TRUE( !has_var( from_hex( 3, "77" ), 2 ) );

  for ( auto i = 0; i < 8; ++i )
  {
    EXPECT_EQ( has_var( from_hex<8>( "3333333333cc33cc3333333333cc33cc33cc33cccccccccc33cc33cccccccccc" ), i ), i % 2 == 1 );
  }

  for ( auto i = 0; i < 8; ++i )
  {
    EXPECT_EQ( has_var( from_hex( 8, "3333333333cc33cc3333333333cc33cc33cc33cccccccccc33cc33cccccccccc" ), i ), i % 2 == 1 );
  }
}

TEST_F( OperationsTest, next )
{
  EXPECT_EQ( next( from_hex<3>( "00" ) ), from_hex<3>( "01" ) );
  EXPECT_EQ( next( from_hex<3>( "ab" ) ), from_hex<3>( "ac" ) );
  EXPECT_EQ( next( from_hex<3>( "ff" ) ), from_hex<3>( "00" ) );

  EXPECT_EQ( next( from_hex( 3, "00" ) ), from_hex( 3, "01" ) );
  EXPECT_EQ( next( from_hex( 3, "ab" ) ), from_hex( 3, "ac" ) );
  EXPECT_EQ( next( from_hex( 3, "ff" ) ), from_hex( 3, "00" ) );

  EXPECT_EQ( next( from_hex<7>( "00000000000000000000000000000000" ) ), from_hex<7>( "00000000000000000000000000000001" ) );
  EXPECT_EQ( next( from_hex<7>( "aaaaaaaaaaaaaaaabbbbbbbbbbbbbbbb" ) ), from_hex<7>( "aaaaaaaaaaaaaaaabbbbbbbbbbbbbbbc" ) );
  EXPECT_EQ( next( from_hex<7>( "ffffffffffffffffffffffffffffffff" ) ), from_hex<7>( "00000000000000000000000000000000" ) );

  EXPECT_EQ( next( from_hex( 7, "00000000000000000000000000000000" ) ), from_hex( 7, "00000000000000000000000000000001" ) );
  EXPECT_EQ( next( from_hex( 7, "aaaaaaaaaaaaaaaabbbbbbbbbbbbbbbb" ) ), from_hex( 7, "aaaaaaaaaaaaaaaabbbbbbbbbbbbbbbc" ) );
  EXPECT_EQ( next( from_hex( 7, "ffffffffffffffffffffffffffffffff" ) ), from_hex( 7, "00000000000000000000000000000000" ) );
}

TEST_F( OperationsTest, cofactors )
{
  EXPECT_EQ( cofactor0( from_hex<3>( "e8" ), 2 ), from_hex<3>( "88" ) );
  EXPECT_EQ( cofactor0( from_hex<3>( "e8" ), 1 ), from_hex<3>( "a0" ) );
  EXPECT_EQ( cofactor0( from_hex<3>( "e8" ), 0 ), from_hex<3>( "c0" ) );

  EXPECT_EQ( cofactor0( from_hex( 3, "e8" ), 2 ), from_hex( 3, "88" ) );
  EXPECT_EQ( cofactor0( from_hex( 3, "e8" ), 1 ), from_hex( 3, "a0" ) );
  EXPECT_EQ( cofactor0( from_hex( 3, "e8" ), 0 ), from_hex( 3, "c0" ) );

  // Co-factors of [(<abc>d){<efg>h}]
  EXPECT_EQ( cofactor0( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 0 ), from_hex<8>( "3fff3fff3fff3fff3fff3fff3fff3fff3fff3fff3fffc0003fffc000c000c000" ) );
  EXPECT_EQ( cofactor0( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 1 ), from_hex<8>( "5fff5fff5fff5fff5fff5fff5fff5fff5fff5fff5fffa0005fffa000a000a000" ) );
  EXPECT_EQ( cofactor0( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 2 ), from_hex<8>( "77ff77ff77ff77ff77ff77ff77ff77ff77ff77ff77ff880077ff880088008800" ) );
  EXPECT_EQ( cofactor0( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 3 ), from_hex<8>( "ffffffffffffffffffffffffffffffffffffffffffff0000ffff000000000000" ) );
  EXPECT_EQ( cofactor0( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 4 ), from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe800e800e800e800e800e800" ) );
  EXPECT_EQ( cofactor0( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 5 ), from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800e800e800" ) );
  EXPECT_EQ( cofactor0( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 6 ), from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ffe800e800e80017ffe800e800e800" ) );
  EXPECT_EQ( cofactor0( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 7 ), from_hex<8>( "17ff17ff17ffe80017ffe800e800e80017ff17ff17ffe80017ffe800e800e800" ) );

  EXPECT_EQ( cofactor0( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 0 ), from_hex( 8, "3fff3fff3fff3fff3fff3fff3fff3fff3fff3fff3fffc0003fffc000c000c000" ) );
  EXPECT_EQ( cofactor0( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 1 ), from_hex( 8, "5fff5fff5fff5fff5fff5fff5fff5fff5fff5fff5fffa0005fffa000a000a000" ) );
  EXPECT_EQ( cofactor0( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 2 ), from_hex( 8, "77ff77ff77ff77ff77ff77ff77ff77ff77ff77ff77ff880077ff880088008800" ) );
  EXPECT_EQ( cofactor0( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 3 ), from_hex( 8, "ffffffffffffffffffffffffffffffffffffffffffff0000ffff000000000000" ) );
  EXPECT_EQ( cofactor0( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 4 ), from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe800e800e800e800e800e800" ) );
  EXPECT_EQ( cofactor0( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 5 ), from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800e800e800" ) );
  EXPECT_EQ( cofactor0( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 6 ), from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ffe800e800e80017ffe800e800e800" ) );
  EXPECT_EQ( cofactor0( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 7 ), from_hex( 8, "17ff17ff17ffe80017ffe800e800e80017ff17ff17ffe80017ffe800e800e800" ) );

  EXPECT_EQ( cofactor1( from_hex<3>( "e8" ), 2 ), from_hex<3>( "ee" ) );
  EXPECT_EQ( cofactor1( from_hex<3>( "e8" ), 1 ), from_hex<3>( "fa" ) );
  EXPECT_EQ( cofactor1( from_hex<3>( "e8" ), 0 ), from_hex<3>( "fc" ) );

  EXPECT_EQ( cofactor1( from_hex( 3, "e8" ), 2 ), from_hex( 3, "ee" ) );
  EXPECT_EQ( cofactor1( from_hex( 3, "e8" ), 1 ), from_hex( 3, "fa" ) );
  EXPECT_EQ( cofactor1( from_hex( 3, "e8" ), 0 ), from_hex( 3, "fc" ) );

  // Co-factors of [(<abc>d){<efg>h}]
  EXPECT_EQ( cofactor1( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 0 ), from_hex<8>( "03ff03ff03ff03ff03ff03ff03ff03ff03ff03ff03fffc0003fffc00fc00fc00" ) );
  EXPECT_EQ( cofactor1( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 1 ), from_hex<8>( "05ff05ff05ff05ff05ff05ff05ff05ff05ff05ff05fffa0005fffa00fa00fa00" ) );
  EXPECT_EQ( cofactor1( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 2 ), from_hex<8>( "11ff11ff11ff11ff11ff11ff11ff11ff11ff11ff11ffee0011ffee00ee00ee00" ) );
  EXPECT_EQ( cofactor1( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 3 ), from_hex<8>( "17171717171717171717171717171717171717171717e8e81717e8e8e8e8e8e8" ) );
  EXPECT_EQ( cofactor1( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 4 ), from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe800e800" ) );
  EXPECT_EQ( cofactor1( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 5 ), from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800" ) );
  EXPECT_EQ( cofactor1( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 6 ), from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ff17ff17ffe800" ) );
  EXPECT_EQ( cofactor1( from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 7 ), from_hex<8>( "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff" ) );

  EXPECT_EQ( cofactor1( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 0 ), from_hex( 8, "03ff03ff03ff03ff03ff03ff03ff03ff03ff03ff03fffc0003fffc00fc00fc00" ) );
  EXPECT_EQ( cofactor1( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 1 ), from_hex( 8, "05ff05ff05ff05ff05ff05ff05ff05ff05ff05ff05fffa0005fffa00fa00fa00" ) );
  EXPECT_EQ( cofactor1( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 2 ), from_hex( 8, "11ff11ff11ff11ff11ff11ff11ff11ff11ff11ff11ffee0011ffee00ee00ee00" ) );
  EXPECT_EQ( cofactor1( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 3 ), from_hex( 8, "17171717171717171717171717171717171717171717e8e81717e8e8e8e8e8e8" ) );
  EXPECT_EQ( cofactor1( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 4 ), from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe800e800" ) );
  EXPECT_EQ( cofactor1( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 5 ), from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800" ) );
  EXPECT_EQ( cofactor1( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 6 ), from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ff17ff17ffe800" ) );
  EXPECT_EQ( cofactor1( from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ffe80017ffe800e800e800" ), 7 ), from_hex( 8, "17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff17ff" ) );
}

TEST_F( OperationsTest, swap_adjacent_inplace_small )
{
  for ( const auto& p : std::vector<std::pair<unsigned, std::string>>{ { 0u, "bce8" }, { 1u, "e6e8" }, { 2u, "dea8" } } )
  {
    auto tt_s = from_hex<4>( "dae8" );
    swap_adjacent_inplace( tt_s, p.first );
    EXPECT_EQ( tt_s, from_hex<4>( p.second ) );

    auto tt_d = from_hex( 4, "dae8" );
    swap_adjacent_inplace( tt_d, p.first );
    EXPECT_EQ( tt_d, from_hex( 4, p.second ) );
  }
}

TEST_F( OperationsTest, swap_adjacent_small )
{
  for ( const auto& p : std::vector<std::pair<unsigned, std::string>>{ { 0u, "bce8" }, { 1u, "e6e8" }, { 2u, "dea8" } } )
  {
    EXPECT_EQ( swap_adjacent( from_hex<4>( "dae8" ), p.first ), from_hex<4>( p.second ) );
    EXPECT_EQ( swap_adjacent( from_hex( 4, "dae8" ), p.first ), from_hex( 4, p.second ) );
  }
}

class OperationsTestSwap : public OperationsTest, public ::testing::WithParamInterface<std::pair<unsigned, std::string>>
{
};

TEST_P( OperationsTestSwap, swap_adjacent_inplace_large )
{
  auto tt_s = from_hex<9>( "28e3b8d62855c4b787eef391a93b33297856658b6743aa3cc7e11fde4e9cbf7c98b07f5febfff33bc7ad6f551bc4cbc440453e1bbe24f0cb4f268c6771b55eee" );
  swap_adjacent_inplace( tt_s, GetParam().first );
  EXPECT_EQ( tt_s, from_hex<9>( GetParam().second ) );

  auto tt_d = from_hex( 9, "28e3b8d62855c4b787eef391a93b33297856658b6743aa3cc7e11fde4e9cbf7c98b07f5febfff33bc7ad6f551bc4cbc440453e1bbe24f0cb4f268c6771b55eee" );
  swap_adjacent_inplace( tt_d, GetParam().first );
  EXPECT_EQ( tt_d, from_hex( 9, GetParam().second ) );
}

TEST_P( OperationsTestSwap, swap_adjacent_large )
{
  EXPECT_EQ( swap_adjacent( from_hex<9>( "28e3b8d62855c4b787eef391a93b33297856658b6743aa3cc7e11fde4e9cbf7c98b07f5febfff33bc7ad6f551bc4cbc440453e1bbe24f0cb4f268c6771b55eee" ),
                            GetParam().first ),
             from_hex<9>( GetParam().second ) );

  EXPECT_EQ( swap_adjacent( from_hex( 9, "28e3b8d62855c4b787eef391a93b33297856658b6743aa3cc7e11fde4e9cbf7c98b07f5febfff33bc7ad6f551bc4cbc440453e1bbe24f0cb4f268c6771b55eee" ),
                            GetParam().first ),
             from_hex( 9, GetParam().second ) );
}

INSTANTIATE_TEST_SUITE_P( OperationsTestSwapInst,
                          OperationsTestSwap,
                          ::testing::Values( std::make_pair( 0u, "48e5d8b64833a2d787eef591c95d55497836638d6725cc5aa7e11fbe2e9adf7a"
                                                                 "98d07f3fedfff55da7cb6f331da2ada220235e1dde42f0ad2f468a6771d33eee" ),
                                             std::make_pair( 1u, "28cbacd62855d09f93facf85a92f0f296c5659a35b43aa3cd3c937f672b4bf7c"
                                                                 "a48c7f77ebffcf2fd3b97b5527d0e3d040513e27be18cce3731ab05b4d9d76fa" ),
                                             std::make_pair( 2u, "2e83bd862585cb478e7ef931a39b32397586685b6473a3acce711dfe49ecb7fc"
                                                                 "9b8075ffefbff33bca7d65f51cb4ccb4440531ebb2e4fc0b42f686c77b155eee" ),
                                             std::make_pair( 3u, "28b8e3d628c455b787f3ee91a9333b297865568b67aa433cc71fe1de4ebf9c7c"
                                                                 "987fb05febf3ff3bc76fad551bcbc4c4403e451bbef024cb4f8c2667715eb5ee" ),
                                             std::make_pair( 4u, "28e32855b8d6c4b787eea93bf391332978566743658baa3cc7e14e9c1fdebf7c"
                                                                 "98b0ebff7f5ff33bc7ad1bc46f55cbc44045be243e1bf0cb4f2671b58c675eee" ),
                                             std::make_pair( 5u, "28e3b8d687eef3912855c4b7a93b33297856658bc7e11fde6743aa3c4e9cbf7c"
                                                                 "98b07f5fc7ad6f55ebfff33b1bc4cbc440453e1b4f268c67be24f0cb71b55eee" ),
                                             std::make_pair( 6u, "28e3b8d62855c4b77856658b6743aa3c87eef391a93b3329c7e11fde4e9cbf7c"
                                                                 "98b07f5febfff33b40453e1bbe24f0cbc7ad6f551bc4cbc44f268c6771b55eee" ),
                                             std::make_pair( 7u, "28e3b8d62855c4b787eef391a93b332998b07f5febfff33bc7ad6f551bc4cbc4"
                                                                 "7856658b6743aa3cc7e11fde4e9cbf7c40453e1bbe24f0cb4f268c6771b55eee" ) ) );

TEST_F( OperationsTest, flip_inplace_small )
{
  for ( const auto& p : std::vector<std::pair<unsigned, std::string>>{ { 0u, "0b34" }, { 1u, "0dc2" }, { 2u, "7083" }, { 3u, "3807" } } )
  {
    auto tt_s = from_hex<4>( "0738" );
    flip_inplace( tt_s, p.first );
    EXPECT_EQ( tt_s, from_hex<4>( p.second ) );

    auto tt_d = from_hex( 4, "0738" );
    flip_inplace( tt_d, p.first );
    EXPECT_EQ( tt_d, from_hex( 4, p.second ) );
  }
}

TEST_F( OperationsTest, flip_small )
{
  for ( const auto& p : std::vector<std::pair<unsigned, std::string>>{ { 0u, "0b34" }, { 1u, "0dc2" }, { 2u, "7083" }, { 3u, "3807" } } )
  {
    EXPECT_EQ( flip( from_hex<4>( "0738" ), p.first ), from_hex<4>( p.second ) );
    EXPECT_EQ( flip( from_hex( 4, "0738" ), p.first ), from_hex( 4, p.second ) );
  }
}

class OperationsTestFlip : public OperationsTest, public ::testing::WithParamInterface<std::pair<unsigned, std::string>>
{
};

TEST_P( OperationsTestFlip, flip_inplace_large )
{
  auto tt_s = from_hex<9>( "8725ca41421c7bba0ca86e26347847526fc346d7f3e79e76566a9493fbef11e24f74a07643afd946195f6a372757e045f3bca58f110ef00ebf2d81e80ba5679f" );
  flip_inplace( tt_s, GetParam().first );
  EXPECT_EQ( tt_s, from_hex<9>( GetParam().second ) );

  auto tt_d = from_hex( 9, "8725ca41421c7bba0ca86e26347847526fc346d7f3e79e76566a9493fbef11e24f74a07643afd946195f6a372757e045f3bca58f110ef00ebf2d81e80ba5679f" );
  flip_inplace( tt_d, GetParam().first );
  EXPECT_EQ( tt_d, from_hex( 9, GetParam().second ) );
}

TEST_P( OperationsTestFlip, flip_large )
{
  EXPECT_EQ( flip( from_hex<9>( "8725ca41421c7bba0ca86e26347847526fc346d7f3e79e76566a9493fbef11e24f74a07643afd946195f6a372757e045f3bca58f110ef00ebf2d81e80ba5679f" ),
                   GetParam().first ),
             from_hex<9>( GetParam().second ) );
  EXPECT_EQ( flip( from_hex( 9, "8725ca41421c7bba0ca86e26347847526fc346d7f3e79e76566a9493fbef11e24f74a07643afd946195f6a372757e045f3bca58f110ef00ebf2d81e80ba5679f" ),
                   GetParam().first ),
             from_hex( 9, GetParam().second ) );
}

INSTANTIATE_TEST_SUITE_P( OperationsTestFlipInst,
                          OperationsTestFlip,
                          ::testing::Values( std::make_pair( 0u, "4b1ac582812cb7750c549d1938b48ba19fc389ebf3db6db9a9956863f7df22d1"
                                                                 "8fb850b9835fe68926af953b1babd08af37c5a4f220df00d7f1e42d4075a9b6f" ),
                                             std::make_pair( 1u, "2d853a141843deea03a29b89c1d21d589f3c197dfcbd6bd9599a616cfebf44b8"
                                                                 "1fd1a0d91caf7619465f9acd8d5db015fce3a52f440bf00bef8724b20ea59d6f" ),
                                             std::make_pair( 2u, "7852ac1424c1b7abc08ae66243877425f63c647d3f7ee96765a64939bffe112e"
                                                                 "f4470a6734fa9d6491f5a67372750e543fcb5af811e00fe0fbd2188eb05a76f9" ),
                                             std::make_pair( 3u, "258741ca1c42ba7ba80c266e78345247c36fd746e7f3769e6a569394effbe211"
                                                                 "744f76a0af4346d95f19376a572745e0bcf38fa50e110ef02dbfe881a50b9f67" ),
                                             std::make_pair( 4u, "ca4187257bba421c6e260ca84752347846d76fc39e76f3e79493566a11e2fbef"
                                                                 "a0764f74d94643af6a37195fe0452757a58ff3bcf00e110e81e8bf2d679f0ba5" ),
                                             std::make_pair( 5u, "421c7bba8725ca41347847520ca86e26f3e79e766fc346d7fbef11e2566a9493"
                                                                 "43afd9464f74a0762757e045195f6a37110ef00ef3bca58f0ba5679fbf2d81e8" ),
                                             std::make_pair( 6u, "0ca86e26347847528725ca41421c7bba566a9493fbef11e26fc346d7f3e79e76"
                                                                 "195f6a372757e0454f74a07643afd946bf2d81e80ba5679ff3bca58f110ef00e" ),
                                             std::make_pair( 7u, "6fc346d7f3e79e76566a9493fbef11e28725ca41421c7bba0ca86e2634784752"
                                                                 "f3bca58f110ef00ebf2d81e80ba5679f4f74a07643afd946195f6a372757e045" ),
                                             std::make_pair( 8u, "4f74a07643afd946195f6a372757e045f3bca58f110ef00ebf2d81e80ba5679f"
                                                                 "8725ca41421c7bba0ca86e26347847526fc346d7f3e79e76566a9493fbef11e2" ) ) );

TEST_F( OperationsTest, min_base )
{
  {
    auto tt = from_hex<3>( "a0" ); /* (ac) */
    const auto support = min_base_inplace( tt );
    EXPECT_EQ( support, std::vector<uint8_t>( { 0, 2 } ) );
    EXPECT_EQ( tt, from_hex<3>( "88" ) );
    expand_inplace( tt, support );
    EXPECT_EQ( tt, from_hex<3>( "a0" ) );
  }

  {
    auto tt = from_hex<4>( "3c3c" ); /* [bc] */
    const auto support = min_base_inplace( tt );
    EXPECT_EQ( support, std::vector<uint8_t>( { 1, 2 } ) );
    EXPECT_EQ( tt, from_hex<4>( "6666" ) );
    expand_inplace( tt, support );
    EXPECT_EQ( tt, from_hex<4>( "3c3c" ) );
  }

  {
    auto tt = from_hex<7>( "ff55ff55ff55ff555555555555555555" ); /* {!a(dg)} */
    const auto support = min_base_inplace( tt );
    EXPECT_EQ( support, std::vector<uint8_t>( { 0, 3, 6 } ) );
    EXPECT_EQ( tt, from_hex<7>( "d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5" ) );
    expand_inplace( tt, support );
    EXPECT_EQ( tt, from_hex<7>( "ff55ff55ff55ff555555555555555555" ) );
  }
}

TEST_F( OperationsTest, expand )
{
  /* create [(ac){bd}] from (ab) and {ac} */
  auto f_and = from_hex( 2, "8" );
  auto f_or = from_hex( 2, "e" );

  auto f_and_e = extend_to( f_and, 4 );
  auto f_or_e = extend_to( f_or, 4 );

  expand_inplace( f_and_e, std::vector<uint8_t>{ 0, 2 } );
  expand_inplace( f_or_e, std::vector<uint8_t>{ 1, 3 } );

  EXPECT_EQ( f_and_e ^ f_or_e, from_hex( 4, "5f6c" ) );
}

TEST_F( OperationsTest, extend_to )
{
  static_truth_table<9> maj, th_extend;
  static_truth_table<8> th;

  create_majority( maj );
  create_threshold( th, 4 );
  extend_to_inplace( th_extend, th );

  EXPECT_EQ( cofactor0( maj, 8 ), th_extend );

  EXPECT_EQ( extend_to<3>( from_hex<2>( "a" ) ), from_hex<3>( "aa" ) );
  EXPECT_EQ( extend_to<3>( from_hex( 2, "a" ) ), from_hex<3>( "aa" ) );
  EXPECT_EQ( extend_to( from_hex<2>( "a" ), 3 ), from_hex( 3, "aa" ) );
  EXPECT_EQ( extend_to( from_hex( 2, "a" ), 3 ), from_hex( 3, "aa" ) );
}

TEST_F( OperationsTest, extend_to_same )
{
  static_truth_table<9> tt1, tt2;

  create_random( tt1 );
  extend_to_inplace( tt2, tt1 );

  EXPECT_EQ( tt1, tt2 );
}

TEST_F( OperationsTest, shrink_to )
{
  EXPECT_EQ( shrink_to<2>( from_hex<3>( "aa" ) ), from_hex<2>( "a" ) );
  EXPECT_EQ( shrink_to<2>( from_hex<3>( "1a" ) ), from_hex<2>( "a" ) );
  EXPECT_EQ( shrink_to<2>( from_hex<2>( "8" ) ), from_hex<2>( "8" ) );

  EXPECT_EQ( shrink_to<5>( from_hex<7>( "cafecafecafecafecafecafecafecafe" ) ), from_hex<5>( "cafecafe" ) );

  EXPECT_EQ( shrink_to<2>( from_hex( 3, "aa" ) ), from_hex<2>( "a" ) );
  EXPECT_EQ( shrink_to<2>( from_hex( 3, "1a" ) ), from_hex<2>( "a" ) );
  EXPECT_EQ( shrink_to<2>( from_hex( 2, "8" ) ), from_hex<2>( "8" ) );

  EXPECT_EQ( shrink_to<5>( from_hex( 7, "cafecafecafecafecafecafecafecafe" ) ), from_hex<5>( "cafecafe" ) );

  EXPECT_EQ( shrink_to( from_hex( 3, "aa" ), 2 ), from_hex( 2, "a" ) );
  EXPECT_EQ( shrink_to( from_hex( 3, "1a" ), 2 ), from_hex( 2, "a" ) );
  EXPECT_EQ( shrink_to( from_hex( 2, "8" ), 2 ), from_hex( 2, "8" ) );

  EXPECT_EQ( shrink_to( from_hex( 7, "cafecafecafecafecafecafecafecafe" ), 5 ), from_hex( 5, "cafecafe" ) );
}

TEST_F( OperationsTest, shift_left )
{
  EXPECT_EQ( shift_left( from_hex<3>( "e8" ), 1 ), from_hex<3>( "d0" ) );
  EXPECT_EQ( shift_left( from_hex<4>( "cafe" ), 4 ), from_hex<4>( "afe0" ) );
  EXPECT_EQ( shift_left( from_hex<7>( "cafeaffe12345678acabacab91837465" ), 4 ), from_hex<7>( "afeaffe12345678acabacab918374650" ) );
  EXPECT_EQ( shift_left( from_hex( 7, "cafeaffe12345678acabacab91837465" ), 8 ), from_hex( 7, "feaffe12345678acabacab9183746500" ) );
  EXPECT_EQ( shift_left( from_hex( 7, "cafeaffe12345678acabacab91837465" ), 64 ), from_hex( 7, "acabacab918374650000000000000000" ) );
  EXPECT_EQ( shift_left( from_hex( 7, "cafeaffe12345678acabacab91837465" ), 0 ), from_hex( 7, "cafeaffe12345678acabacab91837465" ) );
  EXPECT_EQ( shift_left( from_hex( 7, "cafeaffe12345678acabacab91837465" ), 128 ), from_hex( 7, "00000000000000000000000000000000" ) );
}

TEST_F( OperationsTest, shift_right )
{
  EXPECT_EQ( shift_right( from_hex<3>( "e8" ), 1 ), from_hex<3>( "74" ) );
  EXPECT_EQ( shift_right( from_hex<4>( "cafe" ), 4 ), from_hex<4>( "0caf" ) );
  EXPECT_EQ( shift_right( from_hex<7>( "cafeaffe12345678acabacab91837465" ), 4 ), from_hex<7>( "0cafeaffe12345678acabacab9183746" ) );
  EXPECT_EQ( shift_right( from_hex( 7, "cafeaffe12345678acabacab91837465" ), 8 ), from_hex( 7, "00cafeaffe12345678acabacab918374" ) );
  EXPECT_EQ( shift_right( from_hex( 7, "cafeaffe12345678acabacab91837465" ), 64 ), from_hex( 7, "0000000000000000cafeaffe12345678" ) );
  EXPECT_EQ( shift_right( from_hex( 7, "cafeaffe12345678acabacab91837465" ), 0 ), from_hex( 7, "cafeaffe12345678acabacab91837465" ) );
  EXPECT_EQ( shift_right( from_hex( 7, "cafeaffe12345678acabacab91837465" ), 128 ), from_hex( 7, "00000000000000000000000000000000" ) );
}

TEST_F( OperationsTest, majority7 )
{
  const auto a = nth<7>( 0 );
  const auto b = nth<7>( 1 );
  const auto c = nth<7>( 2 );
  const auto d = nth<7>( 3 );
  const auto e = nth<7>( 4 );
  const auto f = nth<7>( 5 );
  const auto g = nth<7>( 6 );

  static_truth_table<7> maj7;
  create_majority( maj7 );

  auto special_func = []( auto a, auto b, auto c, auto d, auto e, auto f )
  {
    return ternary_majority( ternary_majority( a, b, c ), d, ternary_majority( e, f, ternary_majority( a, b, c ) ) );
  };

  // special_func is symmetric in variables {a, b, c}
  EXPECT_EQ( special_func( a, b, c, d, e, f ), special_func( a, c, b, d, e, f ) );
  EXPECT_EQ( special_func( a, b, c, d, e, f ), special_func( b, a, c, d, e, f ) );
  EXPECT_EQ( special_func( a, b, c, d, e, f ), special_func( b, c, a, d, e, f ) );
  EXPECT_EQ( special_func( a, b, c, d, e, f ), special_func( c, a, b, d, e, f ) );
  EXPECT_EQ( special_func( a, b, c, d, e, f ), special_func( c, b, a, d, e, f ) );

  // special func is symmetric in variables {d, e, f}
  EXPECT_EQ( special_func( a, b, c, d, e, f ), special_func( a, b, c, d, f, e ) );
  EXPECT_EQ( special_func( a, b, c, d, e, f ), special_func( a, b, c, e, d, f ) );
  EXPECT_EQ( special_func( a, b, c, d, e, f ), special_func( a, b, c, e, f, d ) );
  EXPECT_EQ( special_func( a, b, c, d, e, f ), special_func( a, b, c, f, d, e ) );
  EXPECT_EQ( special_func( a, b, c, d, e, f ), special_func( a, b, c, f, e, d ) );

  const auto sf0 = special_func( a, b, c, d, e, f );
  const auto sf1 = special_func( d, e, f, a, b, c );

  const auto one = ~maj7.construct();

  const auto th0 = cofactor0( maj7, 6 ); /* threshold-4 function (hamming weight >= 4) */
  const auto th1 = cofactor1( maj7, 6 ); /* threshold-3 function (hamming weight >= 3) */
  const auto te = th1 & ~th0;            /* =3 function (hamming weight = 3) */

  EXPECT_EQ( ~th0 | ( sf0 & sf1 ), one );
  EXPECT_EQ( th0 | ( ~sf0 | ~sf1 ), one );
  EXPECT_EQ( ~th1 | ( sf0 | sf1 ), one );
  EXPECT_EQ( th1 | ( ~sf0 & ~sf1 ), one );

  const auto factor1 = d ^ e ^ f;
  const auto factor2 = a ^ b ^ c;
  EXPECT_EQ( sf0, ( factor1 & th1 ) ^ ( ~factor1 & th0 ) );
  EXPECT_EQ( sf1, ( factor1 & th0 ) ^ ( ~factor1 & th1 ) );
  EXPECT_EQ( sf0, ( factor2 & th0 ) ^ ( ~factor2 & th1 ) );
  EXPECT_EQ( sf1, ( factor2 & th1 ) ^ ( ~factor2 & th0 ) );

  EXPECT_EQ( sf0, th0 | ( factor1 & te ) );
  EXPECT_EQ( sf1, th0 | ( factor2 & te ) );

  EXPECT_EQ( ternary_majority( sf0, g, sf1 ), maj7 );
}

template<typename TT>
void majority_decomposition_acw( TT& f1, TT& f2 )
{
  /* num_vars is even */
  ASSERT_EQ( f1.num_vars() % 2, 0 );

  /* k */
  auto k = f1.num_vars() >> 1;

  dynamic_truth_table lhs( 2 * k - 1 );
  create_majority( lhs );
  extend_to_inplace( f1, lhs );

  dynamic_truth_table rhs1( 2 * k - 1 );
  dynamic_truth_table rhs2( 2 * k - 1 );

  create_threshold( rhs1, k );
  create_threshold( rhs2, k - 2 );

  auto rhs1_e = f2.construct();
  auto rhs2_e = f2.construct();
  extend_to_inplace( rhs1_e, rhs1 );
  extend_to_inplace( rhs2_e, rhs2 );

  create_nth_var( f2, 2 * k - 1 );

  f2 = rhs1_e | ( f2 & rhs2_e );
}

template<typename TT>
void majority_decomposition_even( TT& f1, TT& f2 )
{
  /* num_vars is even */
  ASSERT_EQ( f1.num_vars() % 2, 0 );

  /* k is odd */
  auto k = f1.num_vars() >> 1;
  ASSERT_EQ( k % 2, 1 );

  create_threshold( f1, k );
  f2 = f1;

  auto eq_k = f1.construct();
  create_equals( eq_k, k );

  auto factor1 = f1.construct();
  auto factor2 = f2.construct();

  for ( auto i = 0u; i < k; ++i )
  {
    auto v1 = f1.construct();
    create_nth_var( v1, i );
    factor1 ^= v1;

    auto v2 = f1.construct();
    create_nth_var( v2, k + i );
    factor2 ^= v2;
  }

  f1 |= eq_k & factor1;
  f2 |= eq_k & factor2;
}

TEST_F( OperationsTest, majority_conjecture_small_acw )
{
  constexpr auto k = 4;
  constexpr auto n = 2 * k + 1;

  static_truth_table<n> maj, f1_e, f2_e;
  static_truth_table<n - 1> f1, f2;

  create_majority( maj );
  majority_decomposition_acw( f1, f2 );

  EXPECT_EQ( ternary_majority( extend_to<n>( f1 ), nth<n>( 2 * k ), extend_to<n>( f2 ) ), maj );
}

TEST_F( OperationsTest, majority_conjecture_small )
{
  constexpr auto k = 3;
  constexpr auto n = 2 * k + 1;

  static_truth_table<n> maj, f1_e, f2_e;
  static_truth_table<n - 1> f1, f2;

  create_majority( maj );
  majority_decomposition_even( f1, f2 );

  EXPECT_EQ( ternary_majority( extend_to<n>( f1 ), nth<n>( 2 * k ), extend_to<n>( f2 ) ), maj );
}

TEST_F( OperationsTest, majority_odd_conjecture )
{
  constexpr auto k = 3;
  constexpr auto n = 2 * k + 1;

  static_truth_table<n> maj;
  create_majority( maj );

  /* create f1 */
  static_truth_table<n - 2> f1_m3;
  create_majority( f1_m3 );
  static_truth_table<n - 1> f1;
  extend_to_inplace( f1, f1_m3 );

  /* create f2 */
  static_truth_table<n - 1> f2, rem;
  create_threshold( f2, k );
  create_equals( rem, k );
  f2 |= ~f1 & rem;

  auto f2_alt = nth( n - 1, n - 3 );
  for ( auto i = 4; i <= n; ++i )
  {
    f2_alt = ternary_majority( nth( n - 1, n - i ), nth( n - 1, n - 2 ), f2_alt );
  }

  static_truth_table<n - 1> f2_alt2_p1, f2_alt2_p2;
  create_threshold( f2_alt2_p1, k );
  create_equals( f2_alt2_p2, k );
  auto f2_alt2 = f2_alt2_p1 | ( f2_alt2_p2 & nth<n - 1>( n - 2 ) );

  EXPECT_EQ( f2, f2_alt2 );

  /* create majority */
  static_truth_table<n> f1_e, f2_e;
  extend_to_inplace( f1_e, f1 );
  extend_to_inplace( f2_e, f2 );

  EXPECT_EQ( maj, ternary_majority( nth<n>( n - 1 ), f1_e, f2_e ) );
}

TEST_F( OperationsTest, copy_truth_tables )
{
  kitty::dynamic_truth_table tt1( 3 ), tt2( 7 );
  kitty::static_truth_table<3> tt3;
  kitty::static_truth_table<7> tt4;

  kitty::create_random( tt1 );
  kitty::create_random( tt2 );
  kitty::create_random( tt3 );
  kitty::create_random( tt4 );

  kitty::dynamic_truth_table tt;

  tt = tt1;
  EXPECT_EQ( tt, tt1 );
  tt = tt2;
  EXPECT_EQ( tt, tt2 );
  tt = tt3;
  EXPECT_TRUE( std::equal( tt.begin(), tt.end(), tt3.begin() ) );
  tt = tt4;
  EXPECT_TRUE( std::equal( tt.begin(), tt.end(), tt4.begin() ) );

  static_truth_table<3> tt_s3;
  tt_s3 = tt1;
  EXPECT_TRUE( std::equal( tt_s3.begin(), tt_s3.end(), tt1.begin() ) );
  tt_s3 = tt3;
  EXPECT_EQ( tt_s3, tt3 );

  static_truth_table<7> tt_s7;
  tt_s7 = tt2;
  EXPECT_TRUE( std::equal( tt_s7.begin(), tt_s7.end(), tt2.begin() ) );
  tt_s7 = tt4;
  EXPECT_EQ( tt_s7, tt4 );
}

TEST_F( OperationsTest, implies )
{
  EXPECT_TRUE( implies( from_hex<3>( "88" ), from_hex<3>( "e8" ) ) );
  EXPECT_TRUE( implies( from_hex<3>( "e8" ), from_hex<3>( "e8" ) ) );
  EXPECT_TRUE( !implies( from_hex<3>( "e8" ), from_hex<3>( "88" ) ) );
}

TEST_F( OperationsTest, mux_var )
{
  std::default_random_engine gen;

  for ( auto n = 3u; n < 12u; ++n )
  {
    kitty::dynamic_truth_table tt( n );
    std::uniform_int_distribution<int> distribution( 0u, n - 1u );

    for ( auto r = 0u; r < 100u; ++r )
    {
      kitty::create_random( tt );
      const auto var = distribution( gen );
      const auto then_ = cofactor1( tt, var );
      const auto else_ = cofactor0( tt, var );
      const auto res = mux_var( var, then_, else_ );

      EXPECT_EQ( res, tt );
    }
  }
}

TEST_F( OperationsTest, shift_with_mask )
{
  EXPECT_EQ( shift_with_mask( from_hex<3>( "88" ), 0b101 ), from_hex<3>( "a0" ) );
  EXPECT_EQ( shift_with_mask( from_hex<6>( "e8e8e8e8e8e8e8e8" ), 0b111000 ), from_hex<6>( "ffffff00ff000000" ) );
  EXPECT_EQ( shift_with_mask( from_hex<6>( "7778777877787778" ), 0b101101 ), from_hex<6>( "5f5f5f5f5fa05fa0" ) );
}

TEST_F( OperationsTest, intersection_is_empty )
{
  EXPECT_TRUE( intersection_is_empty( from_hex<3>( "a2" ), from_hex<3>( "5c" ) ) );
  EXPECT_TRUE( intersection_is_empty( from_hex( 3, "a2" ), from_hex( 3, "5c" ) ) );
  EXPECT_TRUE( !intersection_is_empty( from_hex<3>( "f0" ), from_hex<3>( "5c" ) ) );
  EXPECT_TRUE( !intersection_is_empty( from_hex( 3, "f0" ), from_hex( 3, "5c" ) ) );

  EXPECT_TRUE( intersection_is_empty( from_hex<3>( "a2" ), from_hex<3>( "5c" ), from_hex<3>( "01" ) ) );
  EXPECT_TRUE( intersection_is_empty( from_hex( 3, "a2" ), from_hex( 3, "5c" ), from_hex( 3, "01" ) ) );
  EXPECT_TRUE( !intersection_is_empty( from_hex<3>( "f0" ), from_hex<3>( "cc" ), from_hex<3>( "83" ) ) );
  EXPECT_TRUE( !intersection_is_empty( from_hex( 3, "f0" ), from_hex( 3, "cc" ), from_hex( 3, "83" ) ) );

  bool res = intersection_is_empty<static_truth_table<3>, true, false>( from_hex<3>( "a2" ), from_hex<3>( "ff" ) );
  EXPECT_TRUE( res );
}

TEST_F( OperationsTest, unary_for_ternary )
{
  EXPECT_EQ( unary_not( ternary_truth_table<static_truth_table<2>>( from_hex<2>( "8" ), from_hex<2>( "f" ) ) ), ternary_truth_table<static_truth_table<2>>( from_hex<2>( "7" ), from_hex<2>( "f" ) ) );
  EXPECT_EQ( unary_not( ternary_truth_table<static_truth_table<2>>( from_hex<2>( "8" ), from_hex<2>( "a" ) ) ), ternary_truth_table<static_truth_table<2>>( from_hex<2>( "2" ), from_hex<2>( "a" ) ) );
  EXPECT_EQ( unary_not( ternary_truth_table<static_truth_table<3>>( from_hex<3>( "80" ), from_hex<3>( "ff" ) ) ), ternary_truth_table<static_truth_table<3>>( from_hex<3>( "7f" ), from_hex<3>( "ff" ) ) );
  EXPECT_EQ( unary_not( ternary_truth_table<static_truth_table<3>>( from_hex<3>( "80" ), from_hex<3>( "aa" ) ) ), ternary_truth_table<static_truth_table<3>>( from_hex<3>( "2a" ), from_hex<3>( "aa" ) ) );
}

TEST_F( OperationsTest, binary_for_ternary )
{
  auto tt1 = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "5" ), from_hex<2>( "f" ) );
  auto tt2 = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "3" ), from_hex<2>( "f" ) );
  auto res_or = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "7" ), from_hex<2>( "f" ) );
  auto res_and = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "1" ), from_hex<2>( "f" ) );
  auto res_xor = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "6" ), from_hex<2>( "f" ) );
  auto res_mux = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "1" ), from_hex<2>( "f" ) );
  EXPECT_EQ( binary_or( tt1, tt2 ), res_or );
  EXPECT_EQ( binary_and( tt1, tt2 ), res_and );
  EXPECT_EQ( binary_xor( tt1, tt2 ), res_xor );
  EXPECT_EQ( mux_var( 0, tt1, tt2 ), res_mux );

  tt1 = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "8" ), from_hex<2>( "d" ) );
  tt2 = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "4" ), from_hex<2>( "4" ) );
  res_or = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "c" ), from_hex<2>( "c" ) );
  res_and = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "0" ), from_hex<2>( "5" ) );
  res_xor = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "4" ), from_hex<2>( "4" ) );
  res_mux = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "8" ), from_hex<2>( "c" ) );
  EXPECT_EQ( binary_or( tt1, tt2 ), res_or );
  EXPECT_EQ( binary_and( tt1, tt2 ), res_and );
  EXPECT_EQ( binary_xor( tt1, tt2 ), res_xor );
  EXPECT_EQ( mux_var( 1, tt1, tt2 ), res_mux );

  auto tt11 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "adadadadadadadadadadadadadadadad" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) );
  auto tt22 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "97979797979797979797979797979797" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) );
  auto res_and1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "85858585858585858585858585858585" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) );
  auto res_or1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "bfbfbfbfbfbfbfbfbfbfbfbfbfbfbfbf" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) );
  auto res_xor1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) );
  auto res_mux1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "bdbdbdbdbdbdbdbdbdbdbdbdbdbdbdbd" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) );
  EXPECT_EQ( binary_or( tt11, tt22 ), res_or1 );
  EXPECT_EQ( binary_and( tt11, tt22 ), res_and1 );
  EXPECT_EQ( binary_xor( tt11, tt22 ), res_xor1 );
  EXPECT_EQ( mux_var( 0, tt11, tt22 ), res_mux1 );

  tt11 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a8a" ), from_hex<7>( "9b9b9b9b9b9b9b9b9b9b9b9b9b9b9b9b" ) );
  tt22 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "45454545454545454545454545454545" ), from_hex<7>( "4d4d4d4d4d4d4d4d4d4d4d4d4d4d4d4d" ) );
  res_or1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "cfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcf" ), from_hex<7>( "cfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcf" ) );
  res_and1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "00000000000000000000000000000000" ), from_hex<7>( "19191919191919191919191919191919" ) );
  res_xor1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "09090909090909090909090909090909" ), from_hex<7>( "09090909090909090909090909090909" ) );
  res_mux1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "cfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcf" ), from_hex<7>( "cfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcf" ) );
  EXPECT_EQ( binary_or( tt11, tt22 ), res_or1 );
  EXPECT_EQ( binary_and( tt11, tt22 ), res_and1 );
  EXPECT_EQ( binary_xor( tt11, tt22 ), res_xor1 );
  EXPECT_EQ( mux_var( 0, tt11, tt22 ), res_mux1 );
}

TEST_F( OperationsTest, ternary_for_ternary )
{
  auto tt1 = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "9" ), from_hex<2>( "f" ) );
  auto tt2 = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "d" ), from_hex<2>( "f" ) );
  auto tt3 = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "2" ), from_hex<2>( "f" ) );
  auto res_maj = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "9" ), from_hex<2>( "f" ) );
  auto res_ite = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "b" ), from_hex<2>( "f" ) );
  EXPECT_EQ( ternary_majority( tt1, tt2, tt3 ), res_maj );
  EXPECT_EQ( ternary_ite( tt1, tt2, tt3 ), res_ite );

  tt1 = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "8" ), from_hex<2>( "9" ) );
  tt2 = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "0" ), from_hex<2>( "5" ) );
  tt3 = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "9" ), from_hex<2>( "d" ) );
  res_maj = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "8" ), from_hex<2>( "d" ) );
  res_ite = ternary_truth_table<static_truth_table<2>>( from_hex<2>( "1" ), from_hex<2>( "5" ) );
  EXPECT_EQ( ternary_majority( tt1, tt2, tt3 ), res_maj );
  EXPECT_EQ( ternary_ite( tt1, tt2, tt3 ), res_ite );

  auto tt11 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "97979797979797979797979797979797" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) );
  auto tt22 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "85858585858585858585858585858585" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) );
  auto tt33 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "28282828282828282828282828282828" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) );
  auto res_maj1 = tt22;
  auto res_ite1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "adadadadadadadadadadadadadadadad" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) );
  EXPECT_EQ( ternary_majority( tt11, tt22, tt33 ), res_maj1 );
  EXPECT_EQ( ternary_ite( tt11, tt22, tt33 ), res_ite1 );

  tt11 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1" ), from_hex<7>( "afafafafafafafafafafafafafafafaf" ) );
  tt22 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "13131313131313131313131313131313" ), from_hex<7>( "1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b" ) );
  tt33 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a" ), from_hex<7>( "9e9e9e9e9e9e9e9e9e9e9e9e9e9e9e9e" ) );
  res_maj1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "03030303030303030303030303030303" ), from_hex<7>( "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f" ) );
  res_ite1 = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b" ), from_hex<7>( "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f" ) );
  EXPECT_EQ( ternary_majority( tt11, tt22, tt33 ), res_maj1 );
  EXPECT_EQ( ternary_ite( tt11, tt22, tt33 ), res_ite1 );
}

TEST_F( OperationsTest, predicate_for_ternary )
{
  auto tt1 = ternary_truth_table<dynamic_truth_table>( from_hex( 2, "5" ), from_hex( 2, "f" ) );
  auto tt2 = ternary_truth_table<dynamic_truth_table>( from_hex( 2, "3" ), from_hex( 2, "f" ) );
  EXPECT_FALSE( implies( tt1, tt2 ) );
  EXPECT_TRUE( implies( tt1, tt1 ) );
  EXPECT_TRUE( implies( tt2, tt2 ) );
  EXPECT_TRUE( less_than( tt2, tt1 ) );
  EXPECT_TRUE( has_var( tt1, 0 ) );
  EXPECT_FALSE( has_var( tt1, 1 ) );
  EXPECT_TRUE( has_var( tt2, 1 ) );
  EXPECT_FALSE( has_var( tt2, 0 ) );

  tt1 = ternary_truth_table<dynamic_truth_table>( from_hex( 2, "0" ), from_hex( 2, "6" ) );
  tt2 = ternary_truth_table<dynamic_truth_table>( from_hex( 2, "a" ), from_hex( 2, "a" ) );
  EXPECT_FALSE( implies( tt1, tt2 ) );
  EXPECT_TRUE( less_than( tt1, tt2 ) );
  EXPECT_FALSE( has_var( tt1, 0 ) );
  EXPECT_FALSE( has_var( tt1, 1 ) );
  EXPECT_TRUE( has_var( tt2, 0 ) );
  EXPECT_FALSE( has_var( tt2, 1 ) );

  auto tt11 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2" ), from_hex( 7, "ffffffffffffffffffffffffffffffff" ) );
  auto tt22 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c" ), from_hex( 7, "ffffffffffffffffffffffffffffffff" ) );
  EXPECT_FALSE( implies( tt11, tt22 ) );
  EXPECT_TRUE( less_than( tt22, tt11 ) );
  EXPECT_TRUE( has_var( tt11, 0 ) );
  EXPECT_TRUE( has_var( tt11, 1 ) );
  EXPECT_TRUE( has_var( tt11, 2 ) );
  EXPECT_FALSE( has_var( tt11, 3 ) );
  EXPECT_FALSE( has_var( tt11, 4 ) );
  EXPECT_FALSE( has_var( tt11, 5 ) );
  EXPECT_FALSE( has_var( tt11, 6 ) );
  EXPECT_FALSE( has_var( tt22, 0 ) );
  EXPECT_TRUE( has_var( tt22, 1 ) );
  EXPECT_TRUE( has_var( tt22, 2 ) );
  EXPECT_FALSE( has_var( tt22, 3 ) );
  EXPECT_FALSE( has_var( tt22, 4 ) );
  EXPECT_FALSE( has_var( tt22, 5 ) );
  EXPECT_FALSE( has_var( tt22, 6 ) );

  tt11 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "05050505050505050505050505050505" ), from_hex( 7, "4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f" ) );
  tt22 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "13131313131313131313131313131313" ), from_hex( 7, "b3b3b3b3b3b3b3b3b3b3b3b3b3b3b3b3" ) );
  EXPECT_FALSE( implies( tt11, tt22 ) );
  EXPECT_TRUE( less_than( tt11, tt22 ) );
  EXPECT_TRUE( has_var( tt11, 0 ) );
  EXPECT_FALSE( has_var( tt11, 1 ) );
  EXPECT_TRUE( has_var( tt11, 2 ) );
  EXPECT_FALSE( has_var( tt11, 3 ) );
  EXPECT_FALSE( has_var( tt11, 4 ) );
  EXPECT_FALSE( has_var( tt11, 5 ) );
  EXPECT_FALSE( has_var( tt11, 6 ) );
  EXPECT_TRUE( has_var( tt22, 0 ) );
  EXPECT_TRUE( has_var( tt22, 1 ) );
  EXPECT_TRUE( has_var( tt22, 2 ) );
  EXPECT_FALSE( has_var( tt22, 3 ) );
  EXPECT_FALSE( has_var( tt22, 4 ) );
  EXPECT_FALSE( has_var( tt22, 5 ) );
  EXPECT_FALSE( has_var( tt22, 6 ) );
}

TEST_F( OperationsTest, cofactor_for_ternary )
{
  auto tt1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "5b" ), from_hex( 3, "ff" ) );
  auto cof_a0 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "bb" ), from_hex( 3, "ff" ) );
  auto cof_a1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "55" ), from_hex( 3, "ff" ) );
  auto cof_b0 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "5f" ), from_hex( 3, "ff" ) );
  auto cof_b1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "5a" ), from_hex( 3, "ff" ) );
  auto cof_c0 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "f3" ), from_hex( 3, "ff" ) );
  auto cof_c1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "0f" ), from_hex( 3, "ff" ) );
  EXPECT_EQ( cofactor0( tt1, 2 ), cof_a0 );
  EXPECT_EQ( cofactor1( tt1, 2 ), cof_a1 );
  EXPECT_EQ( cofactor0( tt1, 1 ), cof_b0 );
  EXPECT_EQ( cofactor1( tt1, 1 ), cof_b1 );
  EXPECT_EQ( cofactor0( tt1, 0 ), cof_c0 );
  EXPECT_EQ( cofactor1( tt1, 0 ), cof_c1 );

  tt1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "95" ), from_hex( 3, "d5" ) );
  cof_a0 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "55" ), from_hex( 3, "55" ) );
  cof_a1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "99" ), from_hex( 3, "dd" ) );
  cof_b0 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "55" ), from_hex( 3, "55" ) );
  cof_b1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "a5" ), from_hex( 3, "f5" ) );
  cof_c0 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "3f" ), from_hex( 3, "ff" ) );
  cof_c1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "c0" ), from_hex( 3, "c0" ) );
  EXPECT_EQ( cofactor0( tt1, 2 ), cof_a0 );
  EXPECT_EQ( cofactor1( tt1, 2 ), cof_a1 );
  EXPECT_EQ( cofactor0( tt1, 1 ), cof_b0 );
  EXPECT_EQ( cofactor1( tt1, 1 ), cof_b1 );
  EXPECT_EQ( cofactor0( tt1, 0 ), cof_c0 );
  EXPECT_EQ( cofactor1( tt1, 0 ), cof_c1 );

  auto tt11 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "12121212121212121212121212121212" ), from_hex( 7, "97979797979797979797979797979797" ) );
  auto cof_a00 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "22222222222222222222222222222222" ), from_hex( 7, "77777777777777777777777777777777" ) );
  auto cof_a11 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "11111111111111111111111111111111" ), from_hex( 7, "99999999999999999999999999999999" ) );
  auto cof_b00 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a" ), from_hex( 7, "5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f" ) );
  auto cof_b11 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "00000000000000000000000000000000" ), from_hex( 7, "a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5" ) );
  auto cof_c00 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "30303030303030303030303030303030" ), from_hex( 7, "3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f" ) );
  auto cof_c11 = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "03030303030303030303030303030303" ), from_hex( 7, "c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3" ) );
  EXPECT_EQ( cofactor0( tt11, 6 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 6 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 5 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 5 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 4 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 4 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 3 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 3 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 2 ), cof_a00 );
  EXPECT_EQ( cofactor1( tt11, 2 ), cof_a11 );
  EXPECT_EQ( cofactor0( tt11, 1 ), cof_b00 );
  EXPECT_EQ( cofactor1( tt11, 1 ), cof_b11 );
  EXPECT_EQ( cofactor0( tt11, 0 ), cof_c00 );
  EXPECT_EQ( cofactor1( tt11, 0 ), cof_c11 );
}

TEST_F( OperationsTest, next_for_ternary )
{
  auto tt1 = ternary_truth_table<dynamic_truth_table>( from_hex( 2, "0" ), from_hex( 2, "0" ) );
  std::vector<std::string> sample = { "0000", "1111", "1110", "1101", "1100", "1011", "1010", "1001", "1000", "-111", "-110", "-101", "-100", "--11", "--10", "---1" };
  for ( uint16_t i = 15; i > 0; i-- )
  {
    next_inplace( tt1 );
    EXPECT_EQ( to_binary( tt1 ), sample[i] );
  }
}

TEST_F( OperationsTest, swap_flip_for_ternary )
{
  auto tt1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "2c" ), from_hex( 3, "ff" ) );
  auto sw_ad0 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "4a" ), from_hex( 3, "ff" ) );
  auto sw_ad1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "38" ), from_hex( 3, "ff" ) );
  auto sw = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "64" ), from_hex( 3, "ff" ) );
  auto f0 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "1c" ), from_hex( 3, "ff" ) );
  auto f1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "83" ), from_hex( 3, "ff" ) );
  auto f2 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "c2" ), from_hex( 3, "ff" ) );
  EXPECT_EQ( swap_adjacent( tt1, 0 ), sw_ad0 );
  EXPECT_EQ( swap_adjacent( tt1, 1 ), sw_ad1 );
  EXPECT_EQ( swap( tt1, 0, 2 ), sw );
  EXPECT_EQ( flip( tt1, 0 ), f0 );
  EXPECT_EQ( flip( tt1, 1 ), f1 );
  EXPECT_EQ( flip( tt1, 2 ), f2 );

  tt1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "11" ), from_hex( 3, "3b" ) );
  sw_ad0 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "11" ), from_hex( 3, "5d" ) );
  sw_ad1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "05" ), from_hex( 3, "2f" ) );
  sw = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "03" ), from_hex( 3, "73" ) );
  f0 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "22" ), from_hex( 3, "37" ) );
  f1 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "44" ), from_hex( 3, "ce" ) );
  f2 = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "11" ), from_hex( 3, "b3" ) );
  EXPECT_EQ( swap_adjacent( tt1, 0 ), sw_ad0 );
  EXPECT_EQ( swap_adjacent( tt1, 1 ), sw_ad1 );
  EXPECT_EQ( flip( tt1, 0 ), f0 );
  EXPECT_EQ( flip( tt1, 1 ), f1 );
  EXPECT_EQ( flip( tt1, 2 ), f2 );
  EXPECT_EQ( swap( tt1, 0, 2 ), sw );
}

TEST_F( OperationsTest, min_base_for_ternary )
{
  auto tt = ternary_truth_table<static_truth_table<3>>( from_hex<3>( "c0" ), from_hex<3>( "ff" ) ); /* (ab) */
  auto support = min_base_inplace( tt );
  EXPECT_EQ( support, std::vector<uint8_t>( { 1, 2 } ) );
  EXPECT_EQ( tt, ternary_truth_table<static_truth_table<3>>( from_hex<3>( "88" ), from_hex<3>( "ff" ) ) );
  expand_inplace( tt, support );
  EXPECT_EQ( tt, ternary_truth_table<static_truth_table<3>>( from_hex<3>( "c0" ), from_hex<3>( "ff" ) ) );

  tt = ternary_truth_table<static_truth_table<3>>( from_hex<3>( "c0" ), from_hex<3>( "f0" ) );
  support = min_base_inplace( tt );
  EXPECT_EQ( support, std::vector<uint8_t>( { 1, 2 } ) );
  EXPECT_EQ( tt, ternary_truth_table<static_truth_table<3>>( from_hex<3>( "88" ), from_hex<3>( "cc" ) ) );
  expand_inplace( tt, support );
  EXPECT_EQ( tt, ternary_truth_table<static_truth_table<3>>( from_hex<3>( "c0" ), from_hex<3>( "f0" ) ) );

  auto ttt = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "ffffffffffffffffffffffff00000000" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) ); /* {x7 + x6} */
  support = min_base_inplace( ttt );
  EXPECT_EQ( support, std::vector<uint8_t>( { 5, 6 } ) );
  EXPECT_EQ( ttt, ternary_truth_table<static_truth_table<7>>( from_hex<7>( "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) ) );
  expand_inplace( ttt, support );
  EXPECT_EQ( ttt, ternary_truth_table<static_truth_table<7>>( from_hex<7>( "ffffffffffffffffffffffff00000000" ), from_hex<7>( "ffffffffffffffffffffffffffffffff" ) ) );

  ttt = ternary_truth_table<static_truth_table<7>>( from_hex<7>( "ffffffffffffffffaaaaaaaaaaaaaaaa" ), from_hex<7>( "ffffffffffffffff00000000ffffffff" ) ); /* {x7 + x0} */
  support = min_base_inplace( ttt );
  EXPECT_EQ( support, std::vector<uint8_t>( { 0, 6 } ) );
  EXPECT_EQ( ttt, ternary_truth_table<static_truth_table<7>>( from_hex<7>( "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" ), from_hex<7>( "ccccccccffffffffccccccccffffffff" ) ) );
  expand_inplace( ttt, support );
  EXPECT_EQ( ttt, ternary_truth_table<static_truth_table<7>>( from_hex<7>( "ffffffffffffffffaaaaaaaaaaaaaaaa" ), from_hex<7>( "ffffffffffffffff00000000ffffffff" ) ) );
}

TEST_F( OperationsTest, extend_for_ternary )
{
  auto tt = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "69" ), from_hex( 3, "f0" ) );
  EXPECT_EQ( extend_to( tt, 4 ), ternary_truth_table<dynamic_truth_table>( from_hex( 4, "6969" ), from_hex( 4, "f0f0" ) ) );
  EXPECT_EQ( extend_to( tt, 5 ), ternary_truth_table<dynamic_truth_table>( from_hex( 5, "69696969" ), from_hex( 5, "f0f0f0f0" ) ) );

  auto ttt = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "0123456789abcdeffedcba9876543210" ), from_hex( 7, "9abcd3452de8869212aceeac456789bb" ) );
  EXPECT_EQ( extend_to( ttt, 8 ), ternary_truth_table<dynamic_truth_table>( from_hex( 8, "0123456789abcdeffedcba98765432100123456789abcdeffedcba9876543210" ), from_hex( 8, "9abcd3452de8869212aceeac456789bb9abcd3452de8869212aceeac456789bb" ) ) );
}

TEST_F( OperationsTest, shrink_for_ternary )
{
  auto tt = ternary_truth_table<dynamic_truth_table>( from_hex( 4, "baba" ), from_hex( 4, "8989" ) );
  EXPECT_EQ( shrink_to( tt, 3 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "ba" ), from_hex( 3, "89" ) ) );
  EXPECT_EQ( shrink_to( tt, 2 ), ternary_truth_table<dynamic_truth_table>( from_hex( 2, "a" ), from_hex( 2, "9" ) ) );

  auto ttt = ternary_truth_table<dynamic_truth_table>( from_hex( 8, "0123456789abcdeffedcba98765432100123456789abcdeffedcba9876543210" ), from_hex( 8, "9abcd3452de8869212aceeac456789bb0123456789abcdeffedcba9876543210" ) );
  EXPECT_EQ( shrink_to( ttt, 7 ), ternary_truth_table<dynamic_truth_table>( from_hex( 7, "0123456789abcdeffedcba9876543210" ), from_hex( 7, "0123456789abcdeffedcba9876543210" ) ) );
}

TEST_F( OperationsTest, shift_for_ternary )
{
  auto tt = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "ca" ), from_hex( 3, "cb" ) );
  EXPECT_EQ( shift_left( tt, 4 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "a0" ), from_hex( 3, "bf" ) ) );
  EXPECT_EQ( shift_left( tt, 3 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "50" ), from_hex( 3, "5f" ) ) );
  EXPECT_EQ( shift_left( tt, 2 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "28" ), from_hex( 3, "2f" ) ) );
  EXPECT_EQ( shift_left( tt, 1 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "94" ), from_hex( 3, "97" ) ) );
  EXPECT_EQ( shift_right( tt, 4 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "0c" ), from_hex( 3, "fc" ) ) );
  EXPECT_EQ( shift_right( tt, 3 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "19" ), from_hex( 3, "f9" ) ) );
  EXPECT_EQ( shift_right( tt, 2 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "32" ), from_hex( 3, "f2" ) ) );
  EXPECT_EQ( shift_right( tt, 1 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "65" ), from_hex( 3, "e5" ) ) );

  auto ttt = ternary_truth_table<dynamic_truth_table>( from_hex( 7, "abababababababababababababababab" ), from_hex( 7, "ebebebebebebebebebebebebebebebeb" ) );
  EXPECT_EQ( shift_left( ttt, 4 ), ternary_truth_table<dynamic_truth_table>( from_hex( 7, "bababababababababababababababab0" ), from_hex( 7, "bebebebebebebebebebebebebebebebf" ) ) );
  EXPECT_EQ( shift_left( ttt, 3 ), ternary_truth_table<dynamic_truth_table>( from_hex( 7, "5d5d5d5d5d5d5d5d5d5d5d5d5d5d5d58" ), from_hex( 7, "5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f" ) ) );
  EXPECT_EQ( shift_left( ttt, 2 ), ternary_truth_table<dynamic_truth_table>( from_hex( 7, "aeaeaeaeaeaeaeaeaeaeaeaeaeaeaeac" ), from_hex( 7, "afafafafafafafafafafafafafafafaf" ) ) );
  EXPECT_EQ( shift_left( ttt, 1 ), ternary_truth_table<dynamic_truth_table>( from_hex( 7, "57575757575757575757575757575756" ), from_hex( 7, "d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7" ) ) );
  EXPECT_EQ( shift_right( ttt, 4 ), ternary_truth_table<dynamic_truth_table>( from_hex( 7, "0abababababababababababababababa" ), from_hex( 7, "febebebebebebebebebebebebebebebe" ) ) );
  EXPECT_EQ( shift_right( ttt, 3 ), ternary_truth_table<dynamic_truth_table>( from_hex( 7, "15757575757575757575757575757575" ), from_hex( 7, "fd7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d" ) ) );
  EXPECT_EQ( shift_right( ttt, 2 ), ternary_truth_table<dynamic_truth_table>( from_hex( 7, "2aeaeaeaeaeaeaeaeaeaeaeaeaeaeaea" ), from_hex( 7, "fafafafafafafafafafafafafafafafa" ) ) );
  EXPECT_EQ( shift_right( ttt, 1 ), ternary_truth_table<dynamic_truth_table>( from_hex( 7, "55d5d5d5d5d5d5d5d5d5d5d5d5d5d5d5" ), from_hex( 7, "f5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f5" ) ) );
}

TEST_F( OperationsTest, compose_for_ternary )
{
  auto op = ternary_truth_table<dynamic_truth_table>( from_hex( 2, "c" ), from_hex( 2, "d" ) );
  auto v1 = ternary_truth_table<dynamic_truth_table>( from_hex( 2, "2" ), from_hex( 2, "e" ) );
  auto v2 = ternary_truth_table<dynamic_truth_table>( from_hex( 2, "a" ), from_hex( 2, "f" ) );
  std::vector<ternary_truth_table<dynamic_truth_table>> v = { v1, v2 };
  EXPECT_EQ( compose_truth_table( op, v ), ternary_truth_table<dynamic_truth_table>( from_hex( 2, "2" ), from_hex( 2, "6" ) ) );

  auto op1 = ternary_truth_table<static_truth_table<3>>( from_hex<3>( "20" ), from_hex<3>( "ef" ) );
  auto v11 = ternary_truth_table<static_truth_table<3>>( from_hex<3>( "80" ), from_hex<3>( "df" ) );
  auto v21 = ternary_truth_table<static_truth_table<3>>( from_hex<3>( "35" ), from_hex<3>( "bd" ) );
  auto v31 = ternary_truth_table<static_truth_table<3>>( from_hex<3>( "44" ), from_hex<3>( "75" ) );
  std::vector<ternary_truth_table<static_truth_table<3>>> vector = { v11, v21, v31 };
  EXPECT_EQ( compose_truth_table( op1, vector ), ternary_truth_table<static_truth_table<3>>( from_hex<3>( "00" ), from_hex<3>( "7f" ) ) );

  auto op2 = ternary_truth_table<dynamic_truth_table>( from_hex( 4, "a321" ), from_hex( 4, "af3d" ) );
  auto v12 = ternary_truth_table<dynamic_truth_table>( from_hex( 4, "e182" ), from_hex( 4, "eddf" ) );
  auto v22 = ternary_truth_table<dynamic_truth_table>( from_hex( 4, "3015" ), from_hex( 4, "7b9d" ) );
  auto v32 = ternary_truth_table<dynamic_truth_table>( from_hex( 4, "4348" ), from_hex( 4, "f37b" ) );
  auto v42 = ternary_truth_table<dynamic_truth_table>( from_hex( 4, "4948" ), from_hex( 4, "d97f" ) );
  v = { v12, v22, v32, v42 };
  EXPECT_EQ( compose_truth_table( op2, v ), ternary_truth_table<dynamic_truth_table>( from_hex( 4, "0000" ), from_hex( 4, "4319" ) ) );
}

TEST_F( OperationsTest, shift_mask_for_ternary )
{
  auto tt = ternary_truth_table<dynamic_truth_table>( from_hex( 3, "11" ), from_hex( 3, "7d" ) );
  EXPECT_EQ( shift_with_mask( tt, 0b110 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "03" ), from_hex( 3, "ff" ) ) );
  EXPECT_EQ( shift_with_mask( tt, 0b101 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "05" ), from_hex( 3, "ff" ) ) );
  EXPECT_EQ( shift_with_mask( tt, 0b011 ), ternary_truth_table<dynamic_truth_table>( from_hex( 3, "11" ), from_hex( 3, "ff" ) ) );
}

TEST_F( OperationsTest, unary_for_quaternary )
{
  EXPECT_EQ( unary_not( quaternary_truth_table<static_truth_table<3>>( from_hex<3>( "4c" ), from_hex<3>( "85" ) ) ), quaternary_truth_table<static_truth_table<3>>( from_hex<3>( "85" ), from_hex<3>( "4c" ) ) );
  EXPECT_EQ( unary_not( quaternary_truth_table<static_truth_table<7>>( from_hex<7>( "012a012a012a012a012a012a012a012a" ), from_hex<7>( "4cb14cb14cb14cb14cb14cb14cb14cb1" ) ) ), quaternary_truth_table<static_truth_table<7>>( from_hex<7>( "4cb14cb14cb14cb14cb14cb14cb14cb1" ), from_hex<7>( "012a012a012a012a012a012a012a012a" ) ) );
}

TEST_F( OperationsTest, binary_for_quaternary )
{
  auto tt1 = quaternary_truth_table<static_truth_table<4>>( from_hex<4>( "0ff0" ), from_hex<4>( "0f0f" ) );
  auto tt2 = quaternary_truth_table<static_truth_table<4>>( from_hex<4>( "6666" ), from_hex<4>( "5555" ) );
  auto res_or = quaternary_truth_table<static_truth_table<4>>( from_hex<4>( "27f6" ), from_hex<4>( "0505" ) );
  auto res_and = quaternary_truth_table<static_truth_table<4>>( from_hex<4>( "0660" ), from_hex<4>( "175f" ) );
  auto res_xor = quaternary_truth_table<static_truth_table<4>>( from_hex<4>( "0756" ), from_hex<4>( "0765" ) );
  auto res_mux_0 = quaternary_truth_table<static_truth_table<4>>( from_hex<4>( "4ee4" ), from_hex<4>( "5f5f" ) );
  auto res_mux_1 = quaternary_truth_table<static_truth_table<4>>( from_hex<4>( "2ee2" ), from_hex<4>( "1d1d" ) );
  auto res_mux_2 = quaternary_truth_table<static_truth_table<4>>( from_hex<4>( "06f6" ), from_hex<4>( "0505" ) );
  auto res_mux_3 = quaternary_truth_table<static_truth_table<4>>( from_hex<4>( "0f66" ), from_hex<4>( "0f55" ) );

  EXPECT_EQ( binary_or( tt1, tt2 ), res_or );
  EXPECT_EQ( binary_and( tt1, tt2 ), res_and );
  EXPECT_EQ( binary_xor( tt1, tt2 ), res_xor );
  EXPECT_EQ( mux_var( 0, tt1, tt2 ), res_mux_0 );
  EXPECT_EQ( mux_var( 1, tt1, tt2 ), res_mux_1 );
  EXPECT_EQ( mux_var( 2, tt1, tt2 ), res_mux_2 );
  EXPECT_EQ( mux_var( 3, tt1, tt2 ), res_mux_3 );

  auto ttt1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "0ff00ff00ff00ff00ff00ff00ff00ff0" ), from_hex( 7, "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f" ) );
  auto ttt2 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "66666666666666666666666666666666" ), from_hex( 7, "55555555555555555555555555555555" ) );
  auto res_or1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "27f627f627f627f627f627f627f627f6" ), from_hex( 7, "05050505050505050505050505050505" ) );
  auto res_and1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "06600660066006600660066006600660" ), from_hex( 7, "175f175f175f175f175f175f175f175f" ) );
  auto res_xor1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "07560756075607560756075607560756" ), from_hex( 7, "07650765076507650765076507650765" ) );
  auto res_mux_01 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "4ee44ee44ee44ee44ee44ee44ee44ee4" ), from_hex( 7, "5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f" ) );
  auto res_mux_11 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "2ee22ee22ee22ee22ee22ee22ee22ee2" ), from_hex( 7, "1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d" ) );
  auto res_mux_21 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "06f606f606f606f606f606f606f606f6" ), from_hex( 7, "05050505050505050505050505050505" ) );
  auto res_mux_31 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "0f660f660f660f660f660f660f660f66" ), from_hex( 7, "0f550f550f550f550f550f550f550f55" ) );

  EXPECT_EQ( binary_or( ttt1, ttt2 ), res_or1 );
  EXPECT_EQ( binary_and( ttt1, ttt2 ), res_and1 );
  EXPECT_EQ( binary_xor( ttt1, ttt2 ), res_xor1 );
  EXPECT_EQ( mux_var( 0, ttt1, ttt2 ), res_mux_01 );
  EXPECT_EQ( mux_var( 1, ttt1, ttt2 ), res_mux_11 );
  EXPECT_EQ( mux_var( 2, ttt1, ttt2 ), res_mux_21 );
  EXPECT_EQ( mux_var( 3, ttt1, ttt2 ), res_mux_31 );
}

TEST_F( OperationsTest, ternary_for_quaternary )
{
  auto tt1 = quaternary_truth_table<static_truth_table<6>>( from_hex<6>( "0000ffffffff0000" ), from_hex<6>( "0000ffff0000ffff" ) );
  auto tt2 = quaternary_truth_table<static_truth_table<6>>( from_hex<6>( "0ff00ff00ff00ff0" ), from_hex<6>( "0f0f0f0f0f0f0f0f" ) );
  auto tt3 = quaternary_truth_table<static_truth_table<6>>( from_hex<6>( "6666666666666666" ), from_hex<6>( "5555555555555555" ) );
  auto res_maj = quaternary_truth_table<static_truth_table<6>>( from_hex<6>( "0020077627f60660" ), from_hex<6>( "000107570505175f" ) );
  auto res_ite = quaternary_truth_table<static_truth_table<6>>( from_hex<6>( "042007760ff06666" ), from_hex<6>( "040107570f0f5555" ) );
  EXPECT_EQ( ternary_majority( tt1, tt2, tt3 ), res_maj );
  EXPECT_EQ( ternary_ite( tt1, tt2, tt3 ), res_ite );

  auto ttt1 = quaternary_truth_table<static_truth_table<7>>( from_hex<7>( "0000ffffffff00000000ffffffff0000" ), from_hex<7>( "0000ffff0000ffff0000ffff0000ffff" ) );
  auto ttt2 = quaternary_truth_table<static_truth_table<7>>( from_hex<7>( "0ff00ff00ff00ff00ff00ff00ff00ff0" ), from_hex<7>( "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f" ) );
  auto ttt3 = quaternary_truth_table<static_truth_table<7>>( from_hex<7>( "66666666666666666666666666666666" ), from_hex<7>( "55555555555555555555555555555555" ) );
  auto res_maj1 = quaternary_truth_table<static_truth_table<7>>( from_hex<7>( "0020077627f606600020077627f60660" ), from_hex<7>( "000107570505175f000107570505175f" ) );
  auto res_ite1 = quaternary_truth_table<static_truth_table<7>>( from_hex<7>( "042007760ff06666042007760ff06666" ), from_hex<7>( "040107570f0f5555040107570f0f5555" ) );
  EXPECT_EQ( ternary_majority( ttt1, ttt2, ttt3 ), res_maj1 );
  EXPECT_EQ( ternary_ite( ttt1, ttt2, ttt3 ), res_ite1 );
}

TEST_F( OperationsTest, predicate_for_quaternary )
{
  auto tt1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 2, "c" ), from_hex( 2, "9" ) );
  auto tt2 = quaternary_truth_table<dynamic_truth_table>( from_hex( 2, "e" ), from_hex( 2, "9" ) );
  EXPECT_TRUE( implies( tt1, tt2 ) );
  EXPECT_FALSE( implies( tt1, tt1 ) );
  EXPECT_TRUE( implies( tt2, tt2 ) );
  EXPECT_TRUE( less_than( tt1, tt2 ) );
  EXPECT_TRUE( has_var( tt1, 1 ) );
  EXPECT_TRUE( has_var( tt1, 0 ) );
  EXPECT_TRUE( has_var( tt2, 1 ) );
  EXPECT_TRUE( has_var( tt2, 0 ) );
  EXPECT_TRUE( has_var( tt2, 1 ) );

  set_dont_care( tt2, 0 );
  EXPECT_FALSE( has_var( tt2, 1 ) );
  EXPECT_FALSE( has_var( tt2, 0 ) );

  auto ttt1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "cccccccccccccccccccccccccccccccc" ), from_hex( 7, "19191919191919191919191919191919" ) );
  auto ttt2 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d" ), from_hex( 7, "5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b" ) );
  EXPECT_FALSE( implies( ttt1, ttt2 ) );
  EXPECT_FALSE( implies( ttt1, ttt1 ) );
  EXPECT_FALSE( implies( ttt2, ttt2 ) );
  EXPECT_TRUE( less_than( ttt2, ttt1 ) );
  EXPECT_FALSE( has_var( ttt1, 6 ) );
  EXPECT_FALSE( has_var( ttt1, 5 ) );
  EXPECT_FALSE( has_var( ttt1, 4 ) );
  EXPECT_FALSE( has_var( ttt1, 3 ) );
  EXPECT_FALSE( has_var( ttt1, 2 ) );
  EXPECT_TRUE( has_var( ttt1, 1 ) );
  EXPECT_TRUE( has_var( ttt1, 0 ) );
  EXPECT_FALSE( has_var( ttt2, 6 ) );
  EXPECT_FALSE( has_var( ttt2, 5 ) );
  EXPECT_FALSE( has_var( ttt2, 4 ) );
  EXPECT_FALSE( has_var( ttt2, 3 ) );
  EXPECT_TRUE( has_var( ttt2, 2 ) );
  EXPECT_TRUE( has_var( ttt2, 1 ) );
  EXPECT_TRUE( has_var( ttt2, 0 ) );
}

TEST_F( OperationsTest, cofactor_for_quaternary )
{
  auto tt1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "ac" ), from_hex( 3, "6a" ) );
  auto cof_c0 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "0c" ), from_hex( 3, "c0" ) );
  auto cof_c1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "fc" ), from_hex( 3, "3f" ) );
  auto cof_b0 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "a0" ), from_hex( 3, "aa" ) );
  auto cof_b1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "af" ), from_hex( 3, "5a" ) );
  auto cof_a0 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "cc" ), from_hex( 3, "aa" ) );
  auto cof_a1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "aa" ), from_hex( 3, "66" ) );
  EXPECT_EQ( cofactor0( tt1, 2 ), cof_a0 );
  EXPECT_EQ( cofactor1( tt1, 2 ), cof_a1 );
  EXPECT_EQ( cofactor0( tt1, 1 ), cof_b0 );
  EXPECT_EQ( cofactor1( tt1, 1 ), cof_b1 );
  EXPECT_EQ( cofactor0( tt1, 0 ), cof_c0 );
  EXPECT_EQ( cofactor1( tt1, 0 ), cof_c1 );

  auto tt11 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "4d4d4d4d4d4d4d4d4d4d4d4d4d4d4d4d" ), from_hex( 7, "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1" ) );
  auto cof_c11 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c" ), from_hex( 7, "f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0" ) );
  auto cof_c00 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "cfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcf" ), from_hex( 7, "03030303030303030303030303030303" ) );
  auto cof_b00 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "05050505050505050505050505050505" ), from_hex( 7, "a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5" ) );
  auto cof_b11 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f" ), from_hex( 7, "a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0" ) );
  auto cof_a00 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "dddddddddddddddddddddddddddddddd" ), from_hex( 7, "11111111111111111111111111111111" ) );
  auto cof_a11 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "44444444444444444444444444444444" ), from_hex( 7, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" ) );
  EXPECT_EQ( cofactor0( tt11, 6 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 6 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 5 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 5 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 4 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 4 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 3 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 3 ), tt11 );
  EXPECT_EQ( cofactor0( tt11, 2 ), cof_a00 );
  EXPECT_EQ( cofactor1( tt11, 2 ), cof_a11 );
  EXPECT_EQ( cofactor0( tt11, 1 ), cof_b00 );
  EXPECT_EQ( cofactor1( tt11, 1 ), cof_b11 );
  EXPECT_EQ( cofactor0( tt11, 0 ), cof_c00 );
  EXPECT_EQ( cofactor1( tt11, 0 ), cof_c11 );
}

TEST_F( OperationsTest, next_for_quaternary )
{
  auto tt1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 1, "0" ), from_hex( 1, "3" ) );
  std::vector<std::string> sample = { "00", "xx", "x-", "x1", "x0", "-x", "--", "-1", "-0", "1x", "1-", "11", "10", "0x", "0-", "01" };
  for ( uint16_t i = 15; i > 0; i-- )
  {
    next_inplace( tt1 );
    EXPECT_EQ( to_binary( tt1 ), sample[i] );
  }
}

TEST_F( OperationsTest, swap_flip_for_quaternary )
{
  auto tt1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "72" ), from_hex( 3, "3c" ) );
  auto sw_ad0 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "74" ), from_hex( 3, "5a" ) );
  auto sw_ad1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "4e" ), from_hex( 3, "3c" ) );
  auto sw = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "3a" ), from_hex( 3, "66" ) );
  auto f0 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "b1" ), from_hex( 3, "3c" ) );
  auto f1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "d8" ), from_hex( 3, "c3" ) );
  auto f2 = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "27" ), from_hex( 3, "c3" ) );
  EXPECT_EQ( swap_adjacent( tt1, 0 ), sw_ad0 );
  EXPECT_EQ( swap_adjacent( tt1, 1 ), sw_ad1 );
  EXPECT_EQ( swap( tt1, 0, 2 ), sw );
  EXPECT_EQ( flip( tt1, 0 ), f0 );
  EXPECT_EQ( flip( tt1, 1 ), f1 );
  EXPECT_EQ( flip( tt1, 2 ), f2 );

  auto ttt1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "72727272727272727272727272727272" ), from_hex( 7, "3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c" ) );
  auto sw_ad01 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "74747474747474747474747474747474" ), from_hex( 7, "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a" ) );
  auto sw_ad11 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e" ), from_hex( 7, "3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c" ) );
  auto sw1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a" ), from_hex( 7, "66666666666666666666666666666666" ) );
  auto f01 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1" ), from_hex( 7, "3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c" ) );
  auto f11 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "d8d8d8d8d8d8d8d8d8d8d8d8d8d8d8d8" ), from_hex( 7, "c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3" ) );
  auto f21 = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "27272727272727272727272727272727" ), from_hex( 7, "c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3" ) );
  EXPECT_EQ( swap_adjacent( ttt1, 0 ), sw_ad01 );
  EXPECT_EQ( swap_adjacent( ttt1, 1 ), sw_ad11 );
  EXPECT_EQ( swap( ttt1, 0, 2 ), sw1 );
  EXPECT_EQ( flip( ttt1, 0 ), f01 );
  EXPECT_EQ( flip( ttt1, 1 ), f11 );
  EXPECT_EQ( flip( ttt1, 2 ), f21 );
  EXPECT_EQ( flip( ttt1, 3 ), ttt1 );
  EXPECT_EQ( flip( ttt1, 4 ), ttt1 );
  EXPECT_EQ( flip( ttt1, 5 ), ttt1 );
  EXPECT_EQ( flip( ttt1, 6 ), ttt1 );
}

TEST_F( OperationsTest, min_base_for_quaternary )
{
  auto tt = quaternary_truth_table<static_truth_table<3>>( from_hex<3>( "c7" ), from_hex<3>( "0d" ) ); /* (ab) */
  auto support = min_base_inplace( tt );
  EXPECT_EQ( support, std::vector<uint8_t>( { 1, 2 } ) );
  EXPECT_EQ( tt, quaternary_truth_table<static_truth_table<3>>( from_hex<3>( "9b" ), from_hex<3>( "23" ) ) );
  expand_inplace( tt, support );
  EXPECT_EQ( tt, quaternary_truth_table<static_truth_table<3>>( from_hex<3>( "c7" ), from_hex<3>( "0d" ) ) );

  auto ttt = quaternary_truth_table<static_truth_table<7>>( from_hex<7>( "ffffffffffffffffaaaaaaaaaaaaaaaa" ), from_hex<7>( "00000000000000000000000000000000" ) ); /* {x7 + x0} */
  support = min_base_inplace( ttt );
  EXPECT_EQ( support, std::vector<uint8_t>( { 0, 6 } ) );
  EXPECT_EQ( ttt, quaternary_truth_table<static_truth_table<7>>( from_hex<7>( "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" ), from_hex<7>( "00000000000000000000000000000000" ) ) );
  expand_inplace( ttt, support );
  EXPECT_EQ( ttt, quaternary_truth_table<static_truth_table<7>>( from_hex<7>( "ffffffffffffffffaaaaaaaaaaaaaaaa" ), from_hex<7>( "00000000000000000000000000000000" ) ) );
}

TEST_F( OperationsTest, extend_for_quaternary )
{
  auto tt = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "45" ), from_hex( 3, "d1" ) );
  EXPECT_EQ( extend_to( tt, 4 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "4545" ), from_hex( 4, "d1d1" ) ) );
  EXPECT_EQ( extend_to( tt, 5 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 5, "45454545" ), from_hex( 5, "d1d1d1d1" ) ) );

  auto ttt = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "fedcba98765432100123456789abcdef" ), from_hex( 7, "23cd45abc57839ed580ab09529ba4398" ) );
  EXPECT_EQ( extend_to( ttt, 8 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 8, "fedcba98765432100123456789abcdeffedcba98765432100123456789abcdef" ), from_hex( 8, "23cd45abc57839ed580ab09529ba439823cd45abc57839ed580ab09529ba4398" ) ) );
}

TEST_F( OperationsTest, shrink_for_quaternary )
{
  auto tt = quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "7676" ), from_hex( 4, "a2a2" ) );
  EXPECT_EQ( shrink_to( tt, 3 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "76" ), from_hex( 3, "a2" ) ) );
  EXPECT_EQ( shrink_to( tt, 2 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 2, "6" ), from_hex( 2, "2" ) ) );

  auto ttt = quaternary_truth_table<dynamic_truth_table>( from_hex( 8, "fedcba98765432100123456789abcdeffedcba98765432100123456789abcdef" ), from_hex( 8, "23cd45abc57839ed580ab09529ba439823cd45abc57839ed580ab09529ba4398" ) );
  EXPECT_EQ( shrink_to( ttt, 7 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "fedcba98765432100123456789abcdef" ), from_hex( 7, "23cd45abc57839ed580ab09529ba4398" ) ) );
}

TEST_F( OperationsTest, shift_for_quaternary )
{
  auto tt = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "4d" ), from_hex( 3, "8b" ) );
  EXPECT_EQ( shift_left( tt, 4 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "d0" ), from_hex( 3, "bf" ) ) );
  EXPECT_EQ( shift_left( tt, 3 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "68" ), from_hex( 3, "5f" ) ) );
  EXPECT_EQ( shift_left( tt, 2 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "34" ), from_hex( 3, "2f" ) ) );
  EXPECT_EQ( shift_left( tt, 1 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "9a" ), from_hex( 3, "17" ) ) );
  EXPECT_EQ( shift_right( tt, 4 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "04" ), from_hex( 3, "f8" ) ) );
  EXPECT_EQ( shift_right( tt, 3 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "09" ), from_hex( 3, "f1" ) ) );
  EXPECT_EQ( shift_right( tt, 2 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "13" ), from_hex( 3, "e2" ) ) );
  EXPECT_EQ( shift_right( tt, 1 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "26" ), from_hex( 3, "c5" ) ) );

  auto ttt = quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6" ), from_hex( 7, "4b4b4b4b4b4b4b4b4b4b4b4b4b4b4b4b" ) );
  EXPECT_EQ( shift_left( ttt, 4 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c60" ), from_hex( 7, "b4b4b4b4b4b4b4b4b4b4b4b4b4b4b4bf" ) ) );
  EXPECT_EQ( shift_left( ttt, 3 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "36363636363636363636363636363630" ), from_hex( 7, "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5f" ) ) );
  EXPECT_EQ( shift_left( ttt, 2 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b18" ), from_hex( 7, "2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2f" ) ) );
  EXPECT_EQ( shift_left( ttt, 1 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "8d8d8d8d8d8d8d8d8d8d8d8d8d8d8d8c" ), from_hex( 7, "96969696969696969696969696969697" ) ) );
  EXPECT_EQ( shift_right( ttt, 4 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "0c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c" ), from_hex( 7, "f4b4b4b4b4b4b4b4b4b4b4b4b4b4b4b4" ) ) );
  EXPECT_EQ( shift_right( ttt, 3 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "18d8d8d8d8d8d8d8d8d8d8d8d8d8d8d8" ), from_hex( 7, "e9696969696969696969696969696969" ) ) );
  EXPECT_EQ( shift_right( ttt, 2 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "31b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1" ), from_hex( 7, "d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2" ) ) );
  EXPECT_EQ( shift_right( ttt, 1 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 7, "63636363636363636363636363636363" ), from_hex( 7, "a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5" ) ) );
}

TEST_F( OperationsTest, compose_for_quaternary )
{
  auto op = quaternary_truth_table<dynamic_truth_table>( from_hex( 2, "e" ), from_hex( 2, "2" ) );
  auto v1 = quaternary_truth_table<dynamic_truth_table>( from_hex( 2, "9" ), from_hex( 2, "2" ) );
  auto v2 = quaternary_truth_table<dynamic_truth_table>( from_hex( 2, "4" ), from_hex( 2, "5" ) );
  std::vector<quaternary_truth_table<dynamic_truth_table>> v = { v1, v2 };
  EXPECT_EQ( compose_truth_table( op, v ), quaternary_truth_table<dynamic_truth_table>( from_hex( 2, "9" ), from_hex( 2, "0" ) ) );

  auto op1 = quaternary_truth_table<static_truth_table<3>>( from_hex<3>( "e6" ), from_hex<3>( "9d" ) );
  auto v11 = quaternary_truth_table<static_truth_table<3>>( from_hex<3>( "6d" ), from_hex<3>( "72" ) );
  auto v21 = quaternary_truth_table<static_truth_table<3>>( from_hex<3>( "35" ), from_hex<3>( "ec" ) );
  auto v31 = quaternary_truth_table<static_truth_table<3>>( from_hex<3>( "0a" ), from_hex<3>( "a6" ) );
  std::vector<quaternary_truth_table<static_truth_table<3>>> vector = { v11, v21, v31 };
  EXPECT_EQ( compose_truth_table( op1, vector ), quaternary_truth_table<static_truth_table<3>>( from_hex<3>( "2d" ), from_hex<3>( "b4" ) ) );

  auto op2 = quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "8a3b" ), from_hex( 4, "4525" ) );
  auto v12 = quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "19ae" ), from_hex( 4, "d64d" ) );
  auto v22 = quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "ed63" ), from_hex( 4, "d1ac" ) );
  auto v32 = quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "97c2" ), from_hex( 4, "743a" ) );
  auto v42 = quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "2ad5" ), from_hex( 4, "4e9b" ) );
  v = { v12, v22, v32, v42 };
  EXPECT_EQ( compose_truth_table( op2, v ), quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "4084" ), from_hex( 4, "0088" ) ) );
}

TEST_F( OperationsTest, shift_mask_for_quaternary )
{
  auto tt = quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "8c" ), from_hex( 3, "44" ) );
  auto ttab = shift_with_mask( tt, 0b110 );
  auto ttac = shift_with_mask( tt, 0b101 );
  EXPECT_EQ( ttab, quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "c4" ), from_hex( 3, "0c" ) ) );
  EXPECT_EQ( ttac, quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "b0" ), from_hex( 3, "50" ) ) );
  EXPECT_EQ( shift_with_mask( tt, 0b011 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 3, "8c" ), from_hex( 3, "44" ) ) );

  auto ttt = quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "c142" ), from_hex( 4, "33c3" ) );
  auto tttacd = shift_with_mask( ttt, 0b1011 );
  auto tttbcd = shift_with_mask( ttt, 0b0111 );
  auto tttabd = shift_with_mask( ttt, 0b1101 );
  EXPECT_EQ( tttacd, quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "8918" ), from_hex( 4, "3399" ) ) );
  EXPECT_EQ( tttbcd, quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "8198" ), from_hex( 4, "3939" ) ) );
  EXPECT_EQ( tttabd, quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "a124" ), from_hex( 4, "55a5" ) ) );
  EXPECT_EQ( shift_with_mask( ttt, 0b1110 ), quaternary_truth_table<dynamic_truth_table>( from_hex( 4, "c142" ), from_hex( 4, "33c3" ) ) );
}

TEST_F( OperationsTest, ternary_operations_exploiting_dont_cares )
{
  using TT = kitty::static_truth_table<5u>;
  TT bits, care, evol[5];
  kitty::create_from_hex_string( bits, "F0000000" );
  kitty::create_from_hex_string( care, "F0000000" );
  kitty::create_from_hex_string( evol[0], "F0000000" );
  kitty::create_from_hex_string( evol[1], "F0000000" );
  kitty::create_from_hex_string( evol[2], "FF000000" );
  kitty::create_from_hex_string( evol[3], "FFFF0000" );
  kitty::create_from_hex_string( evol[4], "FFFFFFFF" );
  auto tt = ternary_truth_table<TT>( bits, care );
  std::array<bool, 5u> vars;
  static constexpr bool UseDCs = true;
  for ( int i = 0; i < 5; i++ )
  {
    vars[i] = has_var<TT, UseDCs>( tt, i );
    EXPECT_EQ( vars[i], false );
    EXPECT_EQ( tt._bits, evol[i] );
    EXPECT_EQ( tt._care, evol[i] );
  }
}
