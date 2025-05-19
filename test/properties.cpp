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

#include <kitty/properties.hpp>
#include "utility.hpp"

using namespace kitty;

class PropertiesTest : public kitty::testing::Test
{
};

TEST_F( PropertiesTest, chow_small_example )
{
  auto r = chow_parameters( from_hex<2>( "e" ) );
  EXPECT_EQ( r.first, 3u );
  EXPECT_EQ( r.second.size(), 2u );
  EXPECT_EQ( r.second[0u], 2u );
  EXPECT_EQ( r.second[1u], 2u );
}

TEST_F( PropertiesTest, is_canalizing )
{
  static_truth_table<4> tt;
  uint32_t counter{};

  do
  {
    if ( is_canalizing( tt ) )
    {
      ++counter;
    }

    next_inplace( tt );
  } while ( !is_const0( tt ) );

  EXPECT_EQ( counter, 3514u );
}

TEST_F( PropertiesTest, is_selfdual )
{
  static_truth_table<4> tt;
  uint32_t counter{};

  do
  {
    if ( is_selfdual( tt ) )
    {
      ++counter;
    }

    next_inplace( tt );
  } while ( !is_const0( tt ) );

  EXPECT_EQ( counter, 256u );
}

TEST_F( PropertiesTest, is_trivial )
{
  static_truth_table<4> tt;
  uint32_t counter{};

  do
  {
    if ( is_trivial( tt ) )
    {
      ++counter;
    }

    next_inplace( tt );
  } while ( !is_const0( tt ) );

  EXPECT_EQ( counter, 10u );
}

TEST_F( PropertiesTest, is_normal )
{
  static_truth_table<4> tt;
  uint32_t counter{};

  do
  {
    if ( is_normal( tt ) )
    {
      ++counter;
    }

    next_inplace( tt );
  } while ( !is_const0( tt ) );

  EXPECT_EQ( counter, 32768 );
}

TEST_F( PropertiesTest, is_monotone )
{
  static_truth_table<4> tt;
  uint32_t counter{};

  do
  {
    if ( is_monotone( tt ) )
    {
      ++counter;
    }

    next_inplace( tt );
  } while ( !is_const0( tt ) );

  EXPECT_EQ( counter, 168u );
}

TEST_F( PropertiesTest, is_horn )
{
  static_truth_table<4> tt;
  uint32_t counter{};

  do
  {
    if ( is_horn( tt ) )
    {
      ++counter;
    }

    next_inplace( tt );
  } while ( !is_const0( tt ) );

  EXPECT_EQ( counter, 4960u );
}

TEST_F( PropertiesTest, is_krom )
{
  static_truth_table<4> tt;
  uint32_t counter{};

  do
  {
    if ( is_krom( tt ) )
    {
      ++counter;
    }

    next_inplace( tt );
  } while ( !is_const0( tt ) );

  EXPECT_EQ( counter, 4170u );
}

TEST_F( PropertiesTest, runlength_pattern )
{
  auto xor2 = from_hex<2>( "6" );
  auto xor3 = from_hex<3>( "96" );
  auto xor4 = from_hex<4>( "6996" );

  EXPECT_EQ( runlength_pattern( xor2 ), ( std::vector<uint32_t>{ { 1, 2, 1 } } ) );
  EXPECT_EQ( runlength_pattern( xor3 ), ( std::vector<uint32_t>{ { 1, 2, 1, 1, 2, 1 } } ) );
  EXPECT_EQ( runlength_pattern( xor4 ), ( std::vector<uint32_t>{ { 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1 } } ) );
}

TEST_F( PropertiesTest, polynomial_degree )
{
  EXPECT_EQ( polynomial_degree( from_hex<3>( "80" ) ), 3u );
  EXPECT_EQ( polynomial_degree( from_hex<3>( "f7" ) ), 3u );
  EXPECT_EQ( polynomial_degree( from_hex<3>( "e8" ) ), 2u );
  EXPECT_EQ( polynomial_degree( from_hex<3>( "aa" ) ), 1u );
  EXPECT_EQ( polynomial_degree( from_hex<3>( "ff" ) ), 0u );
  EXPECT_EQ( polynomial_degree( from_hex<3>( "00" ) ), 0u );
}
