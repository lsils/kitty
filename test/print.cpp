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

#include <kitty/karnaugh_map.hpp>
#include <kitty/print.hpp>
#include <kitty/properties.hpp>
#include <kitty/static_truth_table.hpp>

#include "utility.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

using namespace kitty;

class PrintTest : public kitty::testing::Test
{
};

TEST_F( PrintTest, print_binary )
{
  EXPECT_EQ( to_binary( from_hex<0>( "0" ) ), "0" );
  EXPECT_EQ( to_binary( from_hex<0>( "1" ) ), "1" );
  EXPECT_EQ( to_binary( from_hex<1>( "1" ) ), "01" );
  EXPECT_EQ( to_binary( from_hex<1>( "2" ) ), "10" );
  EXPECT_EQ( to_binary( from_hex<2>( "8" ) ), "1000" );
  EXPECT_EQ( to_binary( from_hex<3>( "e8" ) ), "11101000" );
  EXPECT_EQ( to_binary( from_hex<7>( "fffefee8fee8e880fee8e880e8808000" ) ),
             "111111111111111011111110111010001111111011101000111010001000000011"
             "11111011101000111010001000000011101000100000001000000000000000" );
}

TEST_F( PrintTest, print_hex )
{
  EXPECT_EQ( to_hex( from_hex<0>( "0" ) ), "0" );
  EXPECT_EQ( to_hex( from_hex<0>( "1" ) ), "1" );
  EXPECT_EQ( to_hex( from_hex<1>( "1" ) ), "1" );
  EXPECT_EQ( to_hex( from_hex<1>( "2" ) ), "2" );
  EXPECT_EQ( to_hex( from_hex<2>( "8" ) ), "8" );
  EXPECT_EQ( to_hex( from_hex<3>( "e8" ) ), "e8" );
  EXPECT_EQ( to_hex( from_hex<7>( "fffefee8fee8e880fee8e880e8808000" ) ),
             "fffefee8fee8e880fee8e880e8808000" );
}

TEST_F( PrintTest, print_raw )
{
  static_truth_table<8> tt, tt2;

  for ( auto i = 0u; i < 100u; ++i )
  {
    create_random( tt );

    std::stringstream ss;
    print_raw( tt, ss );
    ss.seekg( 0 );

    create_from_raw( tt2, ss );

    EXPECT_EQ( tt, tt2 );
  }
}

TEST_F( PrintTest, xmas_one )
{
  std::stringstream ss;
  static_truth_table<5> maj;
  create_majority( maj );
  print_xmas_tree_for_function( maj, ss );

  EXPECT_EQ( 574u, ss.str().size() );
}

TEST_F( PrintTest, xmas_all )
{
  std::stringstream ss;
  print_xmas_tree_for_functions<dynamic_truth_table>(
      3,
      { { is_krom<dynamic_truth_table>, { 31 } },
        { is_horn<dynamic_truth_table>, { 32 } },
        { []( const dynamic_truth_table& f )
          { return is_horn( f ) && is_krom( f ); },
          { 33 } } },
      ss );
  EXPECT_EQ( 7046u, ss.str().size() );
}

template<uint32_t NumVars>
static void anf_to_expression_test()
{
  static_truth_table<NumVars> tt, tt_test;

  for ( auto i = 0u; i < 100u; ++i )
  {
    create_random( tt );
    const auto anf = detail::algebraic_normal_form( tt );
    const auto expr = anf_to_expression( anf );
    create_from_expression( tt_test, expr );
    EXPECT_EQ( tt, tt_test );
  }
}

TEST_F( PrintTest, anf_to_expression )
{
  anf_to_expression_test<0>();
  anf_to_expression_test<1>();
  anf_to_expression_test<2>();
  anf_to_expression_test<3>();
  anf_to_expression_test<4>();
  anf_to_expression_test<5>();
  anf_to_expression_test<6>();
  anf_to_expression_test<7>();
}

TEST_F( PrintTest, print_kmap )
{
  static_truth_table<4> tt1;
  create_random( tt1 );
  karnaugh_map<static_truth_table<4>> k( tt1 );
  std::vector<uint8_t> col_seq = k.get_col_seq();
  std::vector<uint8_t> row_seq = k.get_row_seq();
  std::ofstream outfile( "result.txt" );
  print_kmap( tt1, outfile );
  outfile.close();
  std::ifstream infile( "result.txt" );
  unsigned vars_rows = log2( row_seq.size() );
  std::string read_bit;
  for ( uint8_t i = 0u; i < row_seq.size() + 1; i++ )
  {
    infile >> read_bit;
  }
  while ( !infile.eof() )
  {
    for ( const auto& i : col_seq )
    {
      for ( const auto& j : row_seq )
      {
        infile >> read_bit;
        EXPECT_EQ( std::stoi( read_bit ), get_bit( tt1, ( i << ( vars_rows ) ) + j ) );
      }
      if ( !infile.eof() )
        infile >> read_bit;
    }
  }
  infile.close();
}
