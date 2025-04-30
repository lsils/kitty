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

#include <kitty/spfd.hpp>
#include <kitty/static_truth_table.hpp>
#include <kitty/dynamic_truth_table.hpp>
#include <kitty/partial_truth_table.hpp>

using namespace kitty;

TEST( SpfdTest, spfd_static_default_constructor )
{
  /* Default constructor for static truth table */
  kitty::spfd<kitty::static_truth_table<4u>, 2u> q;
  EXPECT_EQ( q.is_covered(), true );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 0 );
  EXPECT_EQ( q.count_pairs(), 0 );
}

TEST( SpfdTest, spfd_static_completely_specified )
{
  /* Constructor from onset */
  kitty::static_truth_table<2u> func;
  kitty::create_from_binary_string( func, "1001" );
  kitty::spfd<kitty::static_truth_table<2u>, 2u> q( func );
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 4 );
  EXPECT_EQ( q.count_pairs(), 4 );

  /* Remove some minterm pairs */
  kitty::static_truth_table<2u> other;
  kitty::create_from_binary_string( other, "1000" );
  EXPECT_EQ( q.evaluate( other ), 2u );
  EXPECT_EQ( q.update( other ), true );
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 2 );
  EXPECT_EQ( q.count_pairs(), 2 );

  /* Reset to the initial version */
  q.reset();
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 4 );
  EXPECT_EQ( q.count_pairs(), 4 );

  /* Reset to the other function */
  q.reset( other );
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 3 );
  EXPECT_EQ( q.count_pairs(), 3 );
}

TEST( SpfdTest, spfd_static_incompletely_specified )
{
  /* Constructor from onset and careset */
  kitty::static_truth_table<2u> func;
  kitty::create_from_binary_string( func, "1001" );
  kitty::static_truth_table<2u> care;
  kitty::create_from_binary_string( care, "1110" );
  kitty::spfd<kitty::static_truth_table<2u>, 2u> q( func, care );
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 2 );
  EXPECT_EQ( q.count_pairs(), 2 );

  /* Remove some minterm pairs */
  kitty::static_truth_table<2u> other;
  kitty::create_from_binary_string( other, "1000" );
  EXPECT_EQ( q.evaluate( other ), 0u );
  EXPECT_EQ( q.update( other ), true );
  EXPECT_EQ( q.is_covered(), true );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 0 );
  EXPECT_EQ( q.count_pairs(), 0 );

  /* Reset to the initial version */
  q.reset();
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 2 );
  EXPECT_EQ( q.count_pairs(), 2 );

  /* Reset to the other function */
  q.reset( other, care );
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 2 );
  EXPECT_EQ( q.count_pairs(), 2 );
}

TEST( SpfdTest, spfd_dynamic_completely_specified )
{
  /* Constructor from onset */
  kitty::dynamic_truth_table func( 2u );
  kitty::create_from_binary_string( func, "1001" );
  kitty::spfd<kitty::dynamic_truth_table, 2u> q( func );
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 4 );
  EXPECT_EQ( q.count_pairs(), 4 );

  /* Remove some minterm pairs */
  kitty::dynamic_truth_table other( 2u );
  kitty::create_from_binary_string( other, "1000" );
  EXPECT_EQ( q.evaluate( other ), 2u );
  EXPECT_EQ( q.update( other ), true );
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 2 );
  EXPECT_EQ( q.count_pairs(), 2 );

  /* Reset to the initial version */
  q.reset();
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 4 );
  EXPECT_EQ( q.count_pairs(), 4 );

  /* Reset to the other function */
  q.reset( other );
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 3 );
  EXPECT_EQ( q.count_pairs(), 3 );
}

TEST( SpfdTest, spfd_dynamic_incompletely_specified )
{
  /* Constructor from onset and careset */
  kitty::dynamic_truth_table func( 2u );
  kitty::create_from_binary_string( func, "1001" );
  kitty::dynamic_truth_table care( 2u );
  kitty::create_from_binary_string( care, "1110" );
  kitty::spfd<kitty::dynamic_truth_table, 2u> q( func, care );
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 2 );
  EXPECT_EQ( q.count_pairs(), 2 );

  /* Remove some minterm pairs */
  kitty::dynamic_truth_table other( 2u );
  kitty::create_from_binary_string( other, "1000" );
  EXPECT_EQ( q.evaluate( other ), 0u );
  EXPECT_EQ( q.update( other ), true );
  EXPECT_EQ( q.is_covered(), true );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 0 );
  EXPECT_EQ( q.count_pairs(), 0 );

  /* Reset to the initial version */
  q.reset();
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 2 );
  EXPECT_EQ( q.count_pairs(), 2 );

  /* Reset to the other function */
  q.reset( other, care );
  EXPECT_EQ( q.is_covered(), false );
  EXPECT_EQ( q.is_saturated(), false );
  EXPECT_EQ( q.get_num_pairs(), 2 );
  EXPECT_EQ( q.count_pairs(), 2 );
}