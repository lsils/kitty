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

#include <chrono>
#include <vector>

#include <kitty/kitty.hpp>

/* This example is based on Algorithm 4.2.1 in the PhD thesis
 * "Analysis of Affine Equivalent Boolean Functions for Cryptography"
 * by J.E. Fuller (Queensland University of Technology)
 */

/* compile time constant for the number of variables */
auto constexpr num_vars = 5;

int main()
{
  /* truth table type in this example */
  using truth_table = kitty::static_truth_table<num_vars>;

  const auto start = std::chrono::steady_clock::now();
  std::vector<truth_table> fs( 1 );
  fuller_neighborhood_enumeration( fs, []( const auto& tt ) { return exact_spectral_canonization( tt ); } );
  const auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Found " << fs.size() << " classes in " << duration.count() << " s.\n";
}
