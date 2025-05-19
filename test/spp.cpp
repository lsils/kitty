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
#include <kitty/esop.hpp>
#include <kitty/print.hpp>
#include <kitty/spp.hpp>

using namespace kitty;

TEST( SppTest, simple_example )
{
  std::vector<cube> esop{ cube( "111-" ), cube( "11-1" ) };

  auto p = simple_spp( esop, 4u );
  auto cubes = p.first;
  auto sums = p.second;

  EXPECT_EQ( cubes.size(), 1u );
  EXPECT_EQ( cubes[0u]._mask, 0b10011 );
  EXPECT_EQ( cubes[0u]._bits, 0b10011 );
  EXPECT_EQ( sums.size(), 1u );
  EXPECT_EQ( sums[0u], 0b1100 );

  static_truth_table<4> tt_esop, tt_spp;
  create_from_cubes( tt_esop, esop, true );
  create_from_spp( tt_spp, cubes, sums );

  EXPECT_EQ( tt_esop, tt_spp );
}

TEST( SppTest, random_examples )
{
  for ( auto i = 0u; i < 50u; ++i )
  {
    static_truth_table<4> tt, tt_check;
    create_random( tt );
    const auto esop = esop_from_optimum_pkrm( tt );
    const auto p = simple_spp( esop, tt.num_vars() );
    create_from_spp( tt_check, p.first, p.second );

    EXPECT_EQ( tt, tt_check );
  }
}