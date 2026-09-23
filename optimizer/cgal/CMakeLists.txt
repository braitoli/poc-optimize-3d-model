# Builds the `mesh_repair` helper called by the pipeline's CGAL engine.
#
# CGAL 6 is header-only, so this deliberately does NOT use find_package(CGAL):
# build.sh resolves the three header roots (env var -> installed -> vendored)
# and passes them in. That keeps one resolution order in one readable place.
#
#   cmake -S optimizer/cgal -B optimizer/cgal/build \
#         -DCGAL_INCLUDE_DIR=... -DBOOST_INCLUDE_DIR=... -DEIGEN_INCLUDE_DIR=...

cmake_minimum_required(VERSION 3.16)
project(mesh_repair CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

foreach(dir CGAL_INCLUDE_DIR BOOST_INCLUDE_DIR EIGEN_INCLUDE_DIR)
  if(NOT ${dir})
    message(FATAL_ERROR "${dir} is not set - configure through optimizer/cgal/build.sh")
  endif()
  if(NOT IS_DIRECTORY "${${dir}}")
    message(FATAL_ERROR "${dir} points at a missing directory: ${${dir}}")
  endif()
endforeach()

add_executable(mesh_repair mesh_repair.cpp)

# SYSTEM so the (very noisy) CGAL/Boost/Eigen headers do not drown our warnings.
target_include_directories(mesh_repair SYSTEM PRIVATE
  "${CGAL_INCLUDE_DIR}" "${BOOST_INCLUDE_DIR}" "${EIGEN_INCLUDE_DIR}")

# No compiled GMP/MPFR here: CGAL 6 falls back to Boost.Multiprecision, which is
# header-only and keeps this binary dependency-free.
target_compile_definitions(mesh_repair PRIVATE CGAL_DISABLE_GMP=1 CGAL_EIGEN3_ENABLED=1)
target_compile_options(mesh_repair PRIVATE -Wall -Wextra -Wno-unused-parameter)
