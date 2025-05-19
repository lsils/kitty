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
#include <sstream>

#include <kitty/constructors.hpp>
#include <kitty/partial_truth_table.hpp>
#include <kitty/dynamic_truth_table.hpp>
#include <kitty/static_truth_table.hpp>
#include <kitty/print.hpp>

using namespace kitty;

TEST( PartialTruthTableTest, resize )
{
  partial_truth_table tt1( 10 );
  tt1 = ~tt1;
  EXPECT_EQ( count_ones( tt1 ), 10 );
  tt1.resize( 5 );
  EXPECT_EQ( count_ones( tt1 ), 5 );
  tt1.resize( 128 );
  EXPECT_EQ( count_ones( tt1 ), 5 );
  EXPECT_EQ( tt1.num_blocks(), 2 );
}

TEST( PartialTruthTableTest, add_bit )
{
  partial_truth_table tt1( 100 );
  create_random( tt1 );
  EXPECT_EQ( tt1.num_bits(), 100 );
  EXPECT_EQ( tt1.num_blocks(), 2 );

  auto const bits_before = count_ones( tt1 );
  tt1.add_bit( 0 );
  EXPECT_EQ( tt1.num_bits(), 101 );
  EXPECT_EQ( count_ones( tt1 ), bits_before );
  tt1.add_bit( 1 );
  EXPECT_EQ( tt1.num_bits(), 102 );
  EXPECT_EQ( count_ones( tt1 ), bits_before + 1 );

  auto const string_before = to_binary( tt1 );
  tt1.add_bits( 3, 3 ); /* add 011 */
  EXPECT_EQ( to_binary( tt1 ), "011" + string_before );
  partial_truth_table tt2( 64 );
  create_random( tt2 );
  tt1.add_bits( tt2._bits[0], 64 );
  EXPECT_EQ( to_binary( tt1 ), to_binary( tt2 ) + "011" + string_before );
}

TEST( PartialTruthTableTest, hex_binary )
{
  partial_truth_table tt1( 128 );
  create_from_hex_string( tt1, "fffefee8fee8e880fee8e880e8808000" );
  EXPECT_EQ( to_hex( tt1 ), "fffefee8fee8e880fee8e880e8808000" );
  EXPECT_EQ( to_binary( tt1 ), "11111111111111101111111011101000111111101110100011101000100000001111111011101000111010001000000011101000100000001000000000000000" );

  partial_truth_table tt2( 10 );
  create_from_hex_string( tt2, "1a3" );
  EXPECT_EQ( to_hex( tt2 ), "1a3" );
  EXPECT_EQ( to_binary( tt2 ), "0110100011" );

  partial_truth_table tt3( 16 );
  create_from_hex_string( tt3, "f1a3" );
  EXPECT_EQ( to_hex( tt3 ), "f1a3" );
  EXPECT_EQ( to_binary( tt3 ), "1111000110100011" );

  partial_truth_table tt4( 127 );
  create_from_hex_string( tt4, "7ffefee8fee8e880fee8e880e8808000" );
  EXPECT_EQ( to_hex( tt4 ), "7ffefee8fee8e880fee8e880e8808000" );
  EXPECT_EQ( to_binary( tt4 ), "1111111111111101111111011101000111111101110100011101000100000001111111011101000111010001000000011101000100000001000000000000000" );
}

TEST( PartialTruthTableTest, create_nth_var5 )
{
  static_truth_table<5> tt_s;
  partial_truth_table tt_p( 32 );

  for ( auto i = 0; i <= 4; ++i )
  {
    create_nth_var( tt_s, i );
    create_nth_var( tt_p, i );
    EXPECT_EQ( tt_s._bits, tt_p._bits[0] );
  }
}

TEST( PartialTruthTableTest, operations )
{
  dynamic_truth_table tt1_d( 7 );
  create_random( tt1_d );
  dynamic_truth_table tt2_d( 7 );
  create_random( tt2_d );
  partial_truth_table tt1_p( 128 );
  tt1_p = tt1_d;
  partial_truth_table tt2_p( 128 );
  tt2_p = tt2_d;

  auto const tt3_d = tt1_d & tt2_d;
  auto const tt3_p = tt1_p & tt2_p;

  auto const tt4_d = tt1_d >> 100;
  auto const tt4_p = tt1_p >> 100;

  EXPECT_EQ( ( ~tt1_p )._bits, ( ~tt1_d )._bits );
  EXPECT_EQ( tt3_p._bits, tt3_d._bits );
  EXPECT_EQ( tt4_p._bits, tt4_d._bits );
  EXPECT_TRUE( tt1_p == ~( ~tt1_p ) );
}

TEST( PartialTruthTableTest, erase_bit )
{
  partial_truth_table tt( 100 );
  create_random( tt );
  auto str = to_binary( tt );

  tt.erase_bit_shift( 3 );
  str.erase( tt.num_bits() - 3, 1 );
  EXPECT_EQ( to_binary( tt ), str );

  tt.erase_bit_swap( 10 );
  std::swap( str[0], str[tt.num_bits() - 10] );
  str.erase( 0, 1 );
  EXPECT_EQ( to_binary( tt ), str );

  tt.erase_bit_shift( 95 );
  str.erase( tt.num_bits() - 95, 1 );
  EXPECT_EQ( to_binary( tt ), str );

  tt.erase_bit_swap( 88 );
  std::swap( str[0], str[tt.num_bits() - 88] );
  str.erase( 0, 1 );
  EXPECT_EQ( to_binary( tt ), str );
}
