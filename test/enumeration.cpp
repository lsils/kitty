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

#include <kitty/enumeration.hpp>
#include <kitty/npn.hpp>
#include <kitty/static_truth_table.hpp>
#include <kitty/spectral.hpp>

using namespace kitty;

TEST( EnumerationTest, small_spectral )
{
  std::vector<static_truth_table<4>> functions( 1u );
  fuller_neighborhood_enumeration( functions, []( const auto& tt )
                                   { return exact_spectral_canonization( tt ); } );
  ASSERT_EQ( functions.size(), 8u );
}

TEST( EnumerationTest, small_npn )
{
  std::vector<static_truth_table<4>> functions( 1u );
  fuller_neighborhood_enumeration( functions, []( const auto& tt )
                                   { return std::get<0>( exact_npn_canonization( tt ) ); } );
  ASSERT_EQ( functions.size(), 222u );
}

TEST( EnumerationTest, small_spectral_dynamic )
{
  std::vector<dynamic_truth_table> functions( 1u, dynamic_truth_table( 4u ) );
  fuller_neighborhood_enumeration( functions, []( const auto& tt )
                                   { return exact_spectral_canonization( tt ); } );
  ASSERT_EQ( functions.size(), 8u );
}
