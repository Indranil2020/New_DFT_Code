cmake_minimum_required(VERSION 3.24)

project(TIDES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TIDES_ENABLE_CUDA "Build CUDA WP1 substrate probes when CUDA is available" ON)
option(TIDES_ENABLE_HIP "Build HIP/ROCm backend when available" OFF)

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

add_executable(tides_bloch_phase_tests core/tile/tests/bloch_phase_tests.cpp)
target_link_libraries(tides_bloch_phase_tests PRIVATE tides_core)
add_test(NAME bloch_phase_periodic_tiles COMMAND tides_bloch_phase_tests)

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
    target_link_libraries(tides_cuda_tile PUBLIC tides_core cublas cublasLt)
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

    add_executable(tides_cuda_ozaki_gemm_tests
      core/tile/tests/cuda_ozaki_gemm_tests.cpp)
    target_link_libraries(tides_cuda_ozaki_gemm_tests PRIVATE
      tides_core tides_cuda_tile)
    set_target_properties(tides_cuda_ozaki_gemm_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_ozaki_f64e_gemm_vs_fp64_oracle
             COMMAND tides_cuda_ozaki_gemm_tests)
    set_tests_properties(cuda_ozaki_f64e_gemm_vs_fp64_oracle PROPERTIES
      SKIP_RETURN_CODE 77)

    # WP2 T2.5: GPU tile assembly of S, H0
    add_library(tides_cuda_basis STATIC
      core/basis/two_center.cu
      core/basis/three_center.cu)
    target_link_libraries(tides_cuda_basis PUBLIC tides_basis tides_cuda_tile)
    target_compile_options(tides_cuda_basis PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --allow-unsupported-compiler>)
    set_target_properties(tides_cuda_basis PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)

    add_executable(tides_cuda_two_center_tests
      core/basis/tests/cuda_two_center_tests.cpp)
    target_link_libraries(tides_cuda_two_center_tests PRIVATE
      tides_basis tides_cuda_basis)
    set_target_properties(tides_cuda_two_center_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_two_center_tile_assembly
             COMMAND tides_cuda_two_center_tests)
    set_tests_properties(cuda_two_center_tile_assembly PROPERTIES
      SKIP_RETURN_CODE 77)

    # WP2 T2.5: GPU three-center KB assembly
    add_executable(tides_cuda_three_center_tests
      core/basis/tests/cuda_three_center_tests.cpp)
    target_link_libraries(tides_cuda_three_center_tests PRIVATE
      tides_basis tides_cuda_basis)
    set_target_properties(tides_cuda_three_center_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_three_center_kb_assembly
             COMMAND tides_cuda_three_center_tests)
    set_tests_properties(cuda_three_center_kb_assembly PROPERTIES
      SKIP_RETURN_CODE 77)

    # WP3 T3.2: GPU rho builder
    add_library(tides_cuda_grid STATIC
      core/grid/rho_build.cu
      core/grid/poisson_fft.cu
      core/grid/vmat_build.cu
      core/grid/xc.cu)
    target_link_libraries(tides_cuda_grid PUBLIC tides_grid tides_cuda_tile cufft)
    target_compile_options(tides_cuda_grid PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --allow-unsupported-compiler>)
    set_target_properties(tides_cuda_grid PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)

    add_executable(tides_cuda_rho_build_tests
      core/grid/tests/cuda_rho_build_tests.cpp)
    target_link_libraries(tides_cuda_rho_build_tests PRIVATE
      tides_grid tides_cuda_grid)
    set_target_properties(tides_cuda_rho_build_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_rho_build_vs_cpu
             COMMAND tides_cuda_rho_build_tests)
    set_tests_properties(cuda_rho_build_vs_cpu PROPERTIES
      SKIP_RETURN_CODE 77)

    # WP3 T3.4: GPU cuFFT Poisson solver
    add_executable(tides_cuda_poisson_fft_tests
      core/grid/tests/cuda_poisson_fft_tests.cpp)
    target_link_libraries(tides_cuda_poisson_fft_tests PRIVATE
      tides_grid tides_cuda_grid)
    set_target_properties(tides_cuda_poisson_fft_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_poisson_fft_vs_cpu
             COMMAND tides_cuda_poisson_fft_tests)
    set_tests_properties(cuda_poisson_fft_vs_cpu PROPERTIES
      SKIP_RETURN_CODE 77)

    # WP3 T3.3: GPU v->H adjoint
    add_executable(tides_cuda_vmat_build_tests
      core/grid/tests/cuda_vmat_build_tests.cpp)
    target_link_libraries(tides_cuda_vmat_build_tests PRIVATE
      tides_grid tides_cuda_grid)
    set_target_properties(tides_cuda_vmat_build_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_vmat_build_vs_cpu
             COMMAND tides_cuda_vmat_build_tests)
    set_tests_properties(cuda_vmat_build_vs_cpu PROPERTIES
      SKIP_RETURN_CODE 77)

    # WP3 T3.5: GPU XC functional evaluation
    add_executable(tides_cuda_xc_tests
      core/grid/tests/cuda_xc_tests.cpp)
    target_link_libraries(tides_cuda_xc_tests PRIVATE
      tides_grid tides_cuda_grid)
    set_target_properties(tides_cuda_xc_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_xc_lda_vs_cpu
             COMMAND tides_cuda_xc_tests)
    set_tests_properties(cuda_xc_lda_vs_cpu PROPERTIES
      SKIP_RETURN_CODE 77)

    # WP5 T5.3: GPU SP2 purification
    add_library(tides_cuda_sp2 STATIC
      core/solvers/sp2_submatrix/sp2_gpu.cu)
    target_link_libraries(tides_cuda_sp2 PUBLIC tides_solvers tides_cuda_tile)
    target_compile_options(tides_cuda_sp2 PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --allow-unsupported-compiler>)
    set_target_properties(tides_cuda_sp2 PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)

    add_executable(tides_cuda_sp2_tests
      core/solvers/tests/cuda_sp2_tests.cpp)
    target_link_libraries(tides_cuda_sp2_tests PRIVATE
      tides_solvers tides_cuda_sp2)
    set_target_properties(tides_cuda_sp2_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME cuda_sp2_purification_vs_cpu
             COMMAND tides_cuda_sp2_tests)
    set_tests_properties(cuda_sp2_purification_vs_cpu PROPERTIES
      SKIP_RETURN_CODE 77)
  else()
    message(STATUS "CUDA compiler not found; skipping CUDA WP1 targets")
  endif()
endif()

# --- HIP / ROCm backend (T1.8) ----------------------------------------------

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip.cmake)

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

# FFTW3 linking for O(N log N) Poisson solver (T3.4)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)
target_include_directories(tides_grid INTERFACE ${FFTW3_INCLUDE_DIRS})
target_link_libraries(tides_grid INTERFACE ${FFTW3_LIBRARIES})

# --- libxc linking (T3.5) -----------------------------------------------------
# libxc is installed at /home/niel/src/libxc-7.0.0-install/
set(LIBXC_ROOT "/home/niel/src/libxc-7.0.0-install" CACHE PATH "libxc install prefix")
find_path(LIBXC_INCLUDE_DIR xc.h PATHS ${LIBXC_ROOT}/include REQUIRED)
find_library(LIBXC_LIBRARY NAMES xc PATHS ${LIBXC_ROOT}/lib REQUIRED)
message(STATUS "libxc found: ${LIBXC_LIBRARY} (headers: ${LIBXC_INCLUDE_DIR})")
target_include_directories(tides_grid INTERFACE ${LIBXC_INCLUDE_DIR})
target_link_libraries(tides_grid INTERFACE ${LIBXC_LIBRARY})
list(APPEND CMAKE_BUILD_RPATH ${LIBXC_ROOT}/lib)
list(APPEND CMAKE_INSTALL_RPATH ${LIBXC_ROOT}/lib)

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

add_executable(tides_isf_poisson_tests
  core/grid/tests/isf_poisson_tests.cpp)
target_link_libraries(tides_isf_poisson_tests PRIVATE tides_grid)
add_test(NAME isf_poisson_nonperiodic_bcs COMMAND tides_isf_poisson_tests)

add_executable(tides_grid_ops_tests
  core/grid/tests/grid_ops_tests.cpp)
target_link_libraries(tides_grid_ops_tests PRIVATE tides_grid)
add_test(NAME wp3_rho_adjoint_xc_grid_ops
  COMMAND tides_grid_ops_tests)

# T3.5: libxc PBE/GGA test (CPU, no CUDA required)
add_executable(tides_libxc_pbe_tests
  core/grid/tests/libxc_pbe_tests.cpp)
target_link_libraries(tides_libxc_pbe_tests PRIVATE tides_grid)
add_test(NAME wp3_libxc_pbe_vs_analytical COMMAND tides_libxc_pbe_tests)

add_executable(tides_collinear_spin_xc_tests
  core/grid/tests/collinear_spin_xc_tests.cpp)
target_link_libraries(tides_collinear_spin_xc_tests PRIVATE tides_grid)
add_test(NAME collinear_spin_xc COMMAND tides_collinear_spin_xc_tests)

add_executable(tides_qtt_tests
  core/grid/tests/qtt_tests.cpp)
target_link_libraries(tides_qtt_tests PRIVATE tides_grid)
add_test(NAME qtt_rho_poisson_prototype COMMAND tides_qtt_tests)

# T3.9: ESP / Prolate Ewald backend
add_executable(tides_prolate_ewald_tests
  core/grid/tests/prolate_ewald_tests.cpp)
target_link_libraries(tides_prolate_ewald_tests PRIVATE tides_grid)
add_test(NAME t39_prolate_ewald_esp COMMAND tides_prolate_ewald_tests)

# --- WP4: Mid-range solvers (CPU foundation) ---------------------------------

add_library(tides_solvers INTERFACE)
target_include_directories(tides_solvers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(tides_solvers INTERFACE tides_core)
if(LAPACK_FOUND)
  target_link_libraries(tides_solvers INTERFACE ${LAPACK_LIBRARIES})
  target_compile_definitions(tides_solvers INTERFACE TIDES_HAVE_LAPACK=1)
endif()

add_executable(tides_wp4_tests
  core/solvers/tests/wp4_tests.cpp)
target_link_libraries(tides_wp4_tests PRIVATE tides_solvers)
add_test(NAME wp4_solvers_broker_chfsi_omm COMMAND tides_wp4_tests)

add_executable(tides_cusolver_batched_tests
  core/solvers/tests/cusolver_batched_tests.cpp)
target_link_libraries(tides_cusolver_batched_tests PRIVATE tides_solvers)
add_test(NAME cusolver_syevj_batched COMMAND tides_cusolver_batched_tests)

add_executable(tides_wp5_tests
  core/solvers/tests/wp5_tests.cpp)
target_link_libraries(tides_wp5_tests PRIVATE tides_solvers)
add_test(NAME wp5_sp2_submatrix_foe_fermi_truncation COMMAND tides_wp5_tests)

add_executable(tides_gpu_foe_tests
  core/solvers/tests/gpu_foe_tests.cpp)
target_link_libraries(tides_gpu_foe_tests PRIVATE tides_solvers)
add_test(NAME gpu_foe_r3_dispatch COMMAND tides_gpu_foe_tests)

# --- P3: Physics validation tests ---
add_executable(tides_physics_tests
  core/tests/physics_tests.cpp)
target_link_libraries(tides_physics_tests PRIVATE tides_solvers tides_basis)
if(LAPACK_FOUND)
  target_link_libraries(tides_physics_tests PRIVATE ${LAPACK_LIBRARIES})
endif()
add_test(NAME physics_validation COMMAND tides_physics_tests)

# --- P3.3: GPU regression tests ---
add_executable(tides_gpu_regression_tests
  core/tests/gpu_regression_tests.cpp)
target_link_libraries(tides_gpu_regression_tests PRIVATE
  tides_solvers tides_basis tides_cuda_grid tides_cuda_sp2)
if(LAPACK_FOUND)
  target_link_libraries(tides_gpu_regression_tests PRIVATE ${LAPACK_LIBRARIES})
endif()
set_target_properties(tides_gpu_regression_tests PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME gpu_regression COMMAND tides_gpu_regression_tests)
set_tests_properties(gpu_regression PROPERTIES
  SKIP_RETURN_CODE 77)

# --- P4: Benchmark suite ---
add_executable(tides_benchmark
  core/tests/benchmark.cpp)
target_link_libraries(tides_benchmark PRIVATE
  tides_solvers tides_basis tides_cuda_grid tides_cuda_sp2)
if(LAPACK_FOUND)
  target_link_libraries(tides_benchmark PRIVATE ${LAPACK_LIBRARIES})
endif()
set_target_properties(tides_benchmark PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)

# T5.8: 10⁴-atom single-card benchmark
add_executable(tides_10k_atom_benchmark
  core/solvers/tests/t58_10k_atom_benchmark.cpp)
target_link_libraries(tides_10k_atom_benchmark PRIVATE tides_solvers tides_basis)
if(LAPACK_FOUND)
  target_link_libraries(tides_10k_atom_benchmark PRIVATE ${LAPACK_LIBRARIES})
endif()
add_test(NAME t58_10k_atom_single_card COMMAND tides_10k_atom_benchmark)

# T5.9: Distributed R2/R3 scaling simulation (Phase C)
add_executable(tides_distributed_scaling_tests
  core/parallel/tests/t59_distributed_scaling.cpp)
target_link_libraries(tides_distributed_scaling_tests PRIVATE tides_core)
add_test(NAME t59_distributed_scaling COMMAND tides_distributed_scaling_tests)

# T8.6: MPI + NVSHMEM multi-node (Phase C)
add_executable(tides_mpi_nvshmem_tests
  core/parallel/tests/t86_mpi_nvshmem_tests.cpp)
target_link_libraries(tides_mpi_nvshmem_tests PRIVATE tides_core)
add_test(NAME t86_mpi_nvshmem COMMAND tides_mpi_nvshmem_tests)

# L-BFGS optimizer tests
add_executable(tides_lbfgs_tests
  core/dynamics/tests/lbfgs_tests.cpp)
target_link_libraries(tides_lbfgs_tests PRIVATE tides_scf)
add_test(NAME lbfgs_optimizer COMMAND tides_lbfgs_tests)

# Broyden mixer tests
add_executable(tides_broyden_mixer_tests
  core/scf/tests/broyden_mixer_tests.cpp)
target_link_libraries(tides_broyden_mixer_tests PRIVATE tides_scf)
add_test(NAME broyden_mixer COMMAND tides_broyden_mixer_tests)

# Roofline analysis for GPU kernels
add_executable(tides_roofline_analysis
  bench/roofline_analysis.cpp)
add_test(NAME roofline_analysis COMMAND tides_roofline_analysis)

# --- Per-Engine Test & Profile Suites (E1-E8) ---
add_executable(tides_e1_tile_profile
  tests/per_engine/e1_tile/e1_test_profile.cpp
  core/tile/reduce_f64e.cpp)
target_link_libraries(tides_e1_tile_profile PRIVATE
  tides_cuda_tile)
set_target_properties(tides_e1_tile_profile PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME e1_tile_profile COMMAND tides_e1_tile_profile)

add_executable(tides_e2_basis_profile
  tests/per_engine/e2_basis/e2_test_profile.cpp)
target_link_libraries(tides_e2_basis_profile PRIVATE
  tides_basis tides_cuda_basis)
set_target_properties(tides_e2_basis_profile PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME e2_basis_profile COMMAND tides_e2_basis_profile)

add_executable(tides_e3_grid_profile
  tests/per_engine/e3_grid/e3_test_profile.cpp)
target_link_libraries(tides_e3_grid_profile PRIVATE
  tides_grid tides_cuda_grid)
set_target_properties(tides_e3_grid_profile PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME e3_grid_profile COMMAND tides_e3_grid_profile)

add_executable(tides_e4_solvers_profile
  tests/per_engine/e4_solvers/e4_test_profile.cpp)
target_link_libraries(tides_e4_solvers_profile PRIVATE
  tides_solvers tides_cuda_sp2)
set_target_properties(tides_e4_solvers_profile PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME e4_solvers_profile COMMAND tides_e4_solvers_profile)

add_executable(tides_e5_scf_profile
  tests/per_engine/e5_scf/e5_test_profile.cpp)
target_link_libraries(tides_e5_scf_profile PRIVATE
  tides_scf tides_solvers)
set_target_properties(tides_e5_scf_profile PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME e5_scf_profile COMMAND tides_e5_scf_profile)

add_executable(tides_molecule_driver_tests
  core/scf/tests/molecule_driver_tests.cpp)
target_link_libraries(tides_molecule_driver_tests PRIVATE
  tides_scf tides_grid tides_solvers)
set_target_properties(tides_molecule_driver_tests PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME molecule_driver_end_to_end_scf
  COMMAND tides_molecule_driver_tests)

add_executable(tides_e6_dynamics_profile
  tests/per_engine/e6_dynamics/e6_test_profile.cpp)
target_link_libraries(tides_e6_dynamics_profile PRIVATE
  tides_scf tides_solvers)
set_target_properties(tides_e6_dynamics_profile PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME e6_dynamics_profile COMMAND tides_e6_dynamics_profile)

add_executable(tides_e7_parallel_profile
  tests/per_engine/e7_parallel/e7_test_profile.cpp)
target_link_libraries(tides_e7_parallel_profile PRIVATE
  tides_core)
set_target_properties(tides_e7_parallel_profile PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME e7_parallel_profile COMMAND tides_e7_parallel_profile)

add_executable(tides_e8_hybrids_profile
  tests/per_engine/e8_hybrids/e8_test_profile.cpp)
target_link_libraries(tides_e8_hybrids_profile PRIVATE
  tides_core)
set_target_properties(tides_e8_hybrids_profile PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME e8_hybrids_profile COMMAND tides_e8_hybrids_profile)

add_executable(tides_e9_verification_profile
  tests/per_engine/e9_verification/e9_test_profile.cpp)
target_link_libraries(tides_e9_verification_profile PRIVATE
  tides_core)
set_target_properties(tides_e9_verification_profile PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME e9_verification_profile COMMAND tides_e9_verification_profile)

# --- WP6: SCF, XL-BOMD, forces, dynamics (CPU foundation) -------------------

add_library(tides_scf INTERFACE)
target_include_directories(tides_scf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(tides_scf INTERFACE tides_solvers tides_basis)
if(LAPACK_FOUND)
  target_link_libraries(tides_scf INTERFACE ${LAPACK_LIBRARIES})
  target_compile_definitions(tides_scf INTERFACE TIDES_HAVE_LAPACK=1)
endif()

add_executable(tides_wp6_tests
  core/scf/tests/wp6_tests.cpp)
target_link_libraries(tides_wp6_tests PRIVATE tides_scf)
add_test(NAME wp6_scf_forces_xlbomd_optimizers_stress COMMAND tides_wp6_tests)

add_executable(tides_ewald_tests
  core/scf/tests/ewald_tests.cpp)
target_link_libraries(tides_ewald_tests PRIVATE tides_scf)
add_test(NAME ewald_periodic_ion_ion COMMAND tides_ewald_tests)

add_executable(tides_neb_tests
  core/dynamics/tests/neb_tests.cpp)
target_link_libraries(tides_neb_tests PRIVATE tides_core)
add_test(NAME neb_climbing_image COMMAND tides_neb_tests)

add_executable(tides_md_throughput_bench
  core/dynamics/tests/md_throughput_bench.cpp)
target_link_libraries(tides_md_throughput_bench PRIVATE tides_core)
add_test(NAME md_throughput_vs_anchors COMMAND tides_md_throughput_bench)

# --- WP7: Hybrids, dispersion (CPU foundation) -------------------------------

add_library(tides_hybrids INTERFACE)
target_include_directories(tides_hybrids INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(tides_hybrids INTERFACE tides_scf tides_basis)
if(LAPACK_FOUND)
  target_link_libraries(tides_hybrids INTERFACE ${LAPACK_LIBRARIES})
  target_compile_definitions(tides_hybrids INTERFACE TIDES_HAVE_LAPACK=1)
endif()

add_executable(tides_wp7_tests
  core/hybrids/tests/wp7_tests.cpp)
target_link_libraries(tides_wp7_tests PRIVATE tides_hybrids)
add_test(NAME wp7_hybrids_d3_isdf_ace_paw COMMAND tides_wp7_tests)

add_executable(tides_hse_screening_tests
  core/hybrids/tests/hse_screening_tests.cpp)
target_link_libraries(tides_hse_screening_tests PRIVATE tides_hybrids)
add_test(NAME hse_short_range_screening COMMAND tides_hse_screening_tests)

# --- WP8: Parallel, HPC, packaging (CPU foundation) -------------------------

add_library(tides_parallel_lib INTERFACE)
target_include_directories(tides_parallel_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(tides_parallel_lib INTERFACE tides_core)

add_executable(tides_wp8_tests
  core/parallel/tests/wp8_tests.cpp)
target_link_libraries(tides_wp8_tests PRIVATE tides_parallel_lib)
add_test(NAME wp8_stage_dump_partitioner_halo_packaging COMMAND tides_wp8_tests)

add_executable(tides_multi_gpu_tests
  core/parallel/tests/multi_gpu_tests.cpp)
target_link_libraries(tides_multi_gpu_tests PRIVATE tides_parallel_lib)
add_test(NAME multi_gpu_nccl_data_model COMMAND tides_multi_gpu_tests)

# --- WP9: Verification & benchmarks (the red team) ---------------------------

add_library(tides_verification_lib INTERFACE)
target_include_directories(tides_verification_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(tides_verification_lib INTERFACE tides_parallel_lib tides_scf tides_hybrids)

add_executable(tides_wp9_tests
  core/verification/tests/wp9_tests.cpp)
target_link_libraries(tides_wp9_tests PRIVATE tides_verification_lib)
add_test(NAME wp9_ladder_runner_reference_data_ab_fd_dashboard COMMAND tides_wp9_tests)

add_executable(tides_ab_harness_tests
  core/verification/tests/ab_harness_tests.cpp)
target_link_libraries(tides_ab_harness_tests PRIVATE tides_verification_lib)
add_test(NAME ab_harness_nightly_automation COMMAND tides_ab_harness_tests)

add_executable(tides_competitor_farm_tests
  core/verification/tests/competitor_farm_tests.cpp)
target_link_libraries(tides_competitor_farm_tests PRIVATE tides_verification_lib)
add_test(NAME competitor_farm_containers_parsers COMMAND tides_competitor_farm_tests)

add_executable(tides_regression_dashboard_tests
  core/verification/tests/regression_dashboard_tests.cpp)
target_link_libraries(tides_regression_dashboard_tests PRIVATE tides_verification_lib)
add_test(NAME regression_dashboard_energy_metering COMMAND tides_regression_dashboard_tests)

add_executable(tides_campaign_runner_tests
  core/verification/tests/campaign_runner_tests.cpp)
target_link_libraries(tides_campaign_runner_tests PRIVATE tides_verification_lib)
add_test(NAME campaign_runner_reproducibility_archiver COMMAND tides_campaign_runner_tests)

# --- WP10: API, docs, community, releases -----------------------------------
# T10.1: Nanobind Python bindings (optional — requires nanobind installed)
# When built, produces tides/_native*.so importable from Python.
# Build with: cmake -DBUILD_PYTHON_BINDINGS=ON ..

option(BUILD_PYTHON_BINDINGS "Build Python nanobind bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  # nanobind must be installed via pip: pip install nanobind
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NANOBIND_RESULT)
  if(NANOBIND_RESULT EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(tides_native
      api/python/tides/_native.cpp)
    target_link_libraries(tides_native PRIVATE tides_core tides_scf tides_solvers)
    if(LAPACK_FOUND)
      target_link_libraries(tides_native PRIVATE -lmkl_rt)
    endif()
    set_target_properties(tides_native PROPERTIES
      OUTPUT_NAME "_native"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/api/python/tides")
    message(STATUS "WP10: Python nanobind bindings enabled (tides._native)")
  else()
    message(WARNING "WP10: nanobind not found — Python bindings skipped. Install with: pip install nanobind")
  endif()
endif()
