cmake_minimum_required(VERSION 3.24)

project(TIDES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TIDES_ENABLE_CUDA "Build CUDA WP1 substrate probes when CUDA is available" ON)

add_library(tides_core INTERFACE)
target_include_directories(tides_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(tides_core INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

enable_testing()

add_executable(tides_tilemat_tests core/tile/tests/tilemat_tests.cpp)
target_link_libraries(tides_tilemat_tests PRIVATE tides_core)
add_test(NAME tilemat_analytical_invariants COMMAND tides_tilemat_tests)

add_executable(tides_tilemat_dump_fixture
  core/tile/tests/tilemat_dump_fixture.cpp)
target_link_libraries(tides_tilemat_dump_fixture PRIVATE tides_core)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME tilemat_hdf5_stage_dump_schema
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/core/tile/tests/tilemat_hdf5_roundtrip_test.py
      --fixture $<TARGET_FILE:tides_tilemat_dump_fixture>
      --bridge ${CMAKE_CURRENT_SOURCE_DIR}/tools/tilemat_hdf5.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/tilemat_hdf5_roundtrip)
  set_tests_properties(tilemat_hdf5_stage_dump_schema PROPERTIES
    SKIP_RETURN_CODE 77)
endif()

add_executable(tides_tilemat_layout_probe core/tile/tests/tilemat_layout_probe.cpp)
target_link_libraries(tides_tilemat_layout_probe PRIVATE tides_core)

add_executable(tides_spgemm_filtered_tests core/tile/tests/spgemm_filtered_tests.cpp)
target_link_libraries(tides_spgemm_filtered_tests PRIVATE tides_core)
add_test(NAME spgemm_filtered_analytical_bounds COMMAND tides_spgemm_filtered_tests)

add_executable(tides_spgemm_filter_probe core/tile/tests/spgemm_filter_probe.cpp)
target_link_libraries(tides_spgemm_filter_probe PRIVATE tides_core)

add_executable(tides_precision_tests core/tile/tests/precision_tests.cpp)
target_link_libraries(tides_precision_tests PRIVATE tides_core)
add_test(NAME precision_descriptor_and_ledger_contracts COMMAND tides_precision_tests)

add_executable(tides_ops_tests core/tile/tests/ops_tests.cpp)
target_link_libraries(tides_ops_tests PRIVATE tides_core)
add_test(NAME tile_ops_analytical_contracts COMMAND tides_ops_tests)

add_executable(tides_tile_ops_probe core/tile/tests/tile_ops_probe.cpp)
target_link_libraries(tides_tile_ops_probe PRIVATE tides_core)

add_executable(tides_f64e_reference_tests core/tile/tests/f64e_reference_tests.cpp)
target_link_libraries(tides_f64e_reference_tests PRIVATE tides_core)
add_test(NAME f64e_reference_adversarial_numerics COMMAND tides_f64e_reference_tests)

add_executable(tides_f64e_probe core/tile/tests/f64e_probe.cpp)
target_link_libraries(tides_f64e_probe PRIVATE tides_core)

add_executable(tides_ozaki_slice_tests core/tile/tests/ozaki_slice_tests.cpp)
target_link_libraries(tides_ozaki_slice_tests PRIVATE tides_core)
add_test(NAME ozaki_fp16_slice_planning_contracts
  COMMAND tides_ozaki_slice_tests)

add_executable(tides_ozaki_gemm_probe core/tile/tests/ozaki_gemm_probe.cpp)
target_link_libraries(tides_ozaki_gemm_probe PRIVATE tides_core)

add_executable(tides_deterministic_gauntlet_tests
  core/tile/tests/deterministic_gauntlet_tests.cpp)
target_link_libraries(tides_deterministic_gauntlet_tests PRIVATE tides_core)
add_test(NAME wp1_deterministic_substrate_gauntlet
  COMMAND tides_deterministic_gauntlet_tests)

if(TIDES_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75-virtual 90-virtual CACHE STRING
          "CUDA architectures; PTX targets keep builds runnable on newer GPUs")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    add_library(tides_cuda_tile STATIC
      core/tile/gemm_grouped.cu
      core/tile/ozaki.cu
      core/tile/spgemm_filtered.cu)
    target_link_libraries(tides_cuda_tile PUBLIC tides_core cublas)
    target_compile_options(tides_cuda_tile PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --allow-unsupported-compiler>)
    set_target_properties(tides_cuda_tile PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)

    add_executable(tides_cuda_gemm_tests core/tile/tests/cuda_gemm_tests.cpp)
    target_link_libraries(tides_cuda_gemm_tests PRIVATE tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_gemm_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_grouped_gemm_fp64_oracle COMMAND tides_cuda_gemm_tests)
    set_tests_properties(cuda_grouped_gemm_fp64_oracle PROPERTIES
      SKIP_RETURN_CODE 77)

    add_executable(tides_cuda_gemm_probe core/tile/tests/cuda_gemm_probe.cpp)
    target_link_libraries(tides_cuda_gemm_probe PRIVATE
      tides_core tides_cuda_tile cublas cublasLt)
    set_target_properties(tides_cuda_gemm_probe PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)

    add_executable(tides_cuda_graph_tests core/tile/tests/cuda_graph_tests.cpp)
    target_link_libraries(tides_cuda_graph_tests PRIVATE tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_graph_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_graph_replay_fp64_oracle COMMAND tides_cuda_graph_tests)
    set_tests_properties(cuda_graph_replay_fp64_oracle PROPERTIES
      SKIP_RETURN_CODE 77)

    add_executable(tides_cuda_graph_probe core/tile/tests/cuda_graph_probe.cpp)
    target_link_libraries(tides_cuda_graph_probe PRIVATE tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_graph_probe PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)

    add_executable(tides_cuda_determinism_tests
      core/tile/tests/cuda_determinism_tests.cpp)
    target_link_libraries(tides_cuda_determinism_tests PRIVATE tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_determinism_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_deterministic_substrate_gauntlet
      COMMAND tides_cuda_determinism_tests)
    set_tests_properties(cuda_deterministic_substrate_gauntlet PROPERTIES
      SKIP_RETURN_CODE 77)

    add_executable(tides_cuda_runtime_link_probe
      core/tile/tests/cuda_runtime_link_probe.cpp)
    target_link_libraries(tides_cuda_runtime_link_probe PRIVATE tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_runtime_link_probe PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)

    add_executable(tides_cuda_runtime_reduce_link_probe
      core/tile/tests/cuda_runtime_link_probe.cpp
      core/tile/reduce_f64e.cpp)
    target_link_libraries(tides_cuda_runtime_reduce_link_probe PRIVATE tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_runtime_reduce_link_probe PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)

    add_executable(tides_cuda_reduce_f64e_tests
      core/tile/tests/cuda_reduce_f64e_tests.cpp
      core/tile/reduce_f64e.cpp)
    target_link_libraries(tides_cuda_reduce_f64e_tests PRIVATE tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_reduce_f64e_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_f64e_reduction_analytical_contracts
      COMMAND tides_cuda_reduce_f64e_tests)
    set_tests_properties(cuda_f64e_reduction_analytical_contracts PROPERTIES
      SKIP_RETURN_CODE 77)

    add_executable(tides_cuda_reduce_f64e_probe
      core/tile/tests/cuda_reduce_f64e_probe.cpp
      core/tile/reduce_f64e.cpp)
    target_link_libraries(tides_cuda_reduce_f64e_probe PRIVATE tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_reduce_f64e_probe PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)

    add_executable(tides_cuda_spgemm_tests core/tile/tests/cuda_spgemm_tests.cpp)
    target_link_libraries(tides_cuda_spgemm_tests PRIVATE tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_spgemm_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_filtered_spgemm_fp64_oracle COMMAND tides_cuda_spgemm_tests)
    set_tests_properties(cuda_filtered_spgemm_fp64_oracle PROPERTIES
      SKIP_RETURN_CODE 77)

    add_executable(tides_cuda_spgemm_probe core/tile/tests/cuda_spgemm_probe.cpp)
    target_link_libraries(tides_cuda_spgemm_probe PRIVATE tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_spgemm_probe PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
  else()
    message(STATUS "CUDA compiler not found; skipping CUDA WP1 targets")
  endif()
endif()

# --- WP2: Basis & integrals (CPU foundation) ---------------------------------

# LAPACK/BLAS for the dense symmetric eigensolver used by the atomic radial
# solver (system OpenBLAS+LAPACK). Standard scientific libraries, consistent
# with the proposal's use of HDF5/GoogleTest as external deps.
find_package(LAPACK QUIET)
if(LAPACK_FOUND)
  message(STATUS "WP2: LAPACK found (${LAPACK_LIBRARIES})")
else()
  message(STATUS "WP2: LAPACK not found; radial solver falls back to Jacobi")
endif()

add_library(tides_basis INTERFACE)
target_include_directories(tides_basis INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(tides_basis INTERFACE tides_core)
if(LAPACK_FOUND)
  target_link_libraries(tides_basis INTERFACE ${LAPACK_LIBRARIES})
  target_compile_definitions(tides_basis INTERFACE TIDES_HAVE_LAPACK=1)
endif()

add_executable(tides_atomgen_eigensolver_tests
  core/basis/atomgen/tests/eigensolver_tests.cpp)
target_link_libraries(tides_atomgen_eigensolver_tests PRIVATE tides_core)
add_test(NAME wp2_atomgen_eigensolver_analytical
  COMMAND tides_atomgen_eigensolver_tests)

add_executable(tides_atomgen_hydrogenic_tests
  core/basis/atomgen/tests/hydrogenic_tests.cpp)
target_link_libraries(tides_atomgen_hydrogenic_tests PRIVATE tides_basis)
add_test(NAME wp2_atomgen_hydrogenic_eigenvalues COMMAND tides_atomgen_hydrogenic_tests)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core/basis/atomgen/tests/convergence_probe.cpp)
  add_executable(tides_atomgen_convergence_probe
    core/basis/atomgen/tests/convergence_probe.cpp)
  target_link_libraries(tides_atomgen_convergence_probe PRIVATE tides_basis)
endif()

add_executable(tides_lda_xc_tests
  core/basis/atomgen/tests/lda_xc_tests.cpp)
target_link_libraries(tides_lda_xc_tests PRIVATE tides_basis)
add_test(NAME wp2_lda_xc_vs_libxc COMMAND tides_lda_xc_tests)

add_executable(tides_atomic_lda_tests
  core/basis/atomgen/tests/atomic_lda_tests.cpp)
target_link_libraries(tides_atomic_lda_tests PRIVATE tides_basis)
add_test(NAME wp2_atomic_lda_ne_total_energy COMMAND tides_atomic_lda_tests)

add_executable(tides_pseudo_tests
  core/basis/pseudo/tests/pseudo_tests.cpp)
target_link_libraries(tides_pseudo_tests PRIVATE tides_basis)
add_test(NAME wp2_pseudo_validators_and_ghost_detector
  COMMAND tides_pseudo_tests)

add_executable(tides_two_center_tests
  core/basis/tests/two_center_tests.cpp)
target_link_libraries(tides_two_center_tests PRIVATE tides_basis)
add_test(NAME wp2_two_center_rotation_invariance_and_spline
  COMMAND tides_two_center_tests)

add_executable(tides_derivative_tests
  core/basis/tests/derivative_tests.cpp)
target_link_libraries(tides_derivative_tests PRIVATE tides_basis)
add_test(NAME wp2_derivative_streams_fd_validation
  COMMAND tides_derivative_tests)

add_executable(tides_pyscf_overlap_tests
  core/basis/tests/pyscf_overlap_tests.cpp)
target_link_libraries(tides_pyscf_overlap_tests PRIVATE tides_basis)
add_test(NAME wp2_pyscf_overlap_cross_check COMMAND tides_pyscf_overlap_tests)

add_executable(tides_nao_tests
  core/basis/tests/nao_tests.cpp)
target_link_libraries(tides_nao_tests PRIVATE tides_basis)
add_test(NAME wp2_nao_generation COMMAND tides_nao_tests)

add_executable(tides_wp2_profile
  core/basis/tests/profile_wp2.cpp)
target_link_libraries(tides_wp2_profile PRIVATE tides_basis)

# --- WP3: Grids, Poisson, XC (CPU foundation) ---------------------------------

add_library(tides_grid INTERFACE)
target_include_directories(tides_grid INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(tides_grid INTERFACE tides_basis)

add_executable(tides_dual_grid_tests
  core/grid/tests/dual_grid_tests.cpp)
target_link_libraries(tides_dual_grid_tests PRIVATE tides_grid)
add_test(NAME wp3_dual_grid_index_map_and_halo
  COMMAND tides_dual_grid_tests)

add_executable(tides_poisson_tests
  core/grid/tests/poisson_tests.cpp)
target_link_libraries(tides_poisson_tests PRIVATE tides_grid)
add_test(NAME wp3_poisson_gaussian_charge_all_bcs
  COMMAND tides_poisson_tests)

add_executable(tides_grid_ops_tests
  core/grid/tests/grid_ops_tests.cpp)
target_link_libraries(tides_grid_ops_tests PRIVATE tides_grid)
add_test(NAME wp3_rho_adjoint_xc_grid_ops
  COMMAND tides_grid_ops_tests)
