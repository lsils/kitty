cmake_minimum_required(VERSION 3.1)
project(kitty LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Options
option(KITTY_EXAMPLES "Build examples" ON)
option(KITTY_BENCH "Build benchmarks" OFF)
option(KITTY_TEST "Build tests" OFF)
option(KITTY_TIDY "Check code with clang-tidy" OFF)

# some specific compiler definitions
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fcolor-diagnostics" HAS_FCOLOR_DIAGNOSTICS)
if (HAS_FCOLOR_DIAGNOSTICS)
  add_compile_options(-fcolor-diagnostics)
endif()

# compiler options for SIMD operations
if(MSVC)
  add_compile_options(/arch:AVX2)
elseif(UNIX)
  check_cxx_compiler_flag("-mavx2" HAS_MAVX2)
  if(HAS_MAVX2)
    add_compile_options(-mavx2)
  endif()
endif()

if (UNIX)
  add_compile_options(-W -Wall)
else()
  add_compile_options(/EHsc /MTd)
endif()

add_subdirectory(include)

if(KITTY_EXAMPLES)
  add_subdirectory(examples)
endif()

if(KITTY_BENCH)
  add_subdirectory(bench EXCLUDE_FROM_ALL)
endif()

if(KITTY_TEST)
  add_subdirectory(test)
endif()

if(KITTY_TIDY)
  add_subdirectory(tidy EXCLUDE_FROM_ALL)
endif()

