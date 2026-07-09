# cmake/hip.cmake — HIP / ROCm build support for TIDES tile substrate.
#
# Usage: include this from the main CMakeLists.txt when TIDES_ENABLE_HIP=ON.
# Detects ROCm, configures HIP language, and sets up HIP targets.

option(TIDES_ENABLE_HIP "Build HIP/ROCm backend when available" OFF)

if(TIDES_ENABLE_HIP)
  # Try to find ROCm installation.
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to ROCm installation")
    elseif(DEFINED ENV{ROCM_HOME})
      set(ROCM_PATH $ENV{ROCM_HOME} CACHE PATH "Path to ROCm installation")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
    endif()
  endif()

  # Check for HIP compiler.
  find_program(HIP_HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS ${ROCM_PATH}/bin
    DOC "HIP compiler (hipcc)")

  if(NOT HIP_HIPCC_EXECUTABLE)
    message(STATUS "HIP: hipcc not found in ${ROCM_PATH}/bin — HIP backend disabled")
    set(TIDES_HIP_AVAILABLE OFF)
  else()
    message(STATUS "HIP: Found hipcc: ${HIP_HIPCC_EXECUTABLE}")
    set(TIDES_HIP_AVAILABLE ON)

    # Enable HIP language.
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)

    # Default GPU architectures for AMD (gfx906 = MI50, gfx90a = MI200, gfx1100 = RX 7900).
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
      set(CMAKE_HIP_ARCHITECTURES gfx906 gfx90a gfx1100 CACHE STRING
          "AMD GPU architectures (GCN ISA targets)")
    endif()

    # Find rocBLAS.
    find_path(ROCBLAS_INCLUDE_DIR
      NAMES hipblas.h
      HINTS ${ROCM_PATH}/include)
    find_library(ROCBLAS_LIBRARY
      NAMES hipblas
      HINTS ${ROCM_PATH}/lib)
    if(ROCBLAS_INCLUDE_DIR AND ROCBLAS_LIBRARY)
      message(STATUS "HIP: Found rocBLAS: ${ROCBLAS_LIBRARY}")
      set(TIDES_HAVE_ROCBLAS ON)
    else()
      message(STATUS "HIP: rocBLAS not found — some GPU kernels will be unavailable")
    endif()

    # Find hipSPARSE.
    find_library(HIPSPARSE_LIBRARY
      NAMES hipsparse
      HINTS ${ROCM_PATH}/lib)
    if(HIPSPARSE_LIBRARY)
      message(STATUS "HIP: Found hipSPARSE: ${HIPSPARSE_LIBRARY}")
      set(TIDES_HAVE_HIPSPARSE ON)
    endif()

    # Find rocFFT.
    find_library(ROCFFT_LIBRARY
      NAMES rocfft
      HINTS ${ROCM_PATH}/lib)
    if(ROCFFT_LIBRARY)
      message(STATUS "HIP: Found rocFFT: ${ROCFFT_LIBRARY}")
      set(TIDES_HAVE_ROCFFT ON)
    endif()

    # Find rocSOLVER.
    find_library(ROCSOLVER_LIBRARY
      NAMES rocsolver
      HINTS ${ROCM_PATH}/lib)
    if(ROCSOLVER_LIBRARY)
      message(STATUS "HIP: Found rocSOLVER: ${ROCSOLVER_LIBRARY}")
      set(TIDES_HAVE_ROCSOLVER ON)
    endif()

    # HIP target: tides_hip_tile (mirrors tides_cuda_tile).
    add_library(tides_hip_tile STATIC
      core/tile/gemm_grouped.cu
      core/tile/ozaki.cu
      core/tile/spgemm_filtered.cu)
    target_include_directories(tides_hip_tile PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/core
      ${ROCM_PATH}/include)
    target_link_libraries(tides_hip_tile PUBLIC tides_core)
    target_compile_definitions(tides_hip_tile PUBLIC TIDES_USE_HIP)
    if(TIDES_HAVE_ROCBLAS)
      target_link_libraries(tides_hip_tile PUBLIC ${ROCBLAS_LIBRARY})
    endif()
    if(TIDES_HAVE_HIPSPARSE)
      target_link_libraries(tides_hip_tile PUBLIC ${HIPSPARSE_LIBRARY})
    endif()
    set_target_properties(tides_hip_tile PROPERTIES
      LANGUAGE HIP
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

    # HIP test: basic runtime probe.
    add_executable(tides_hip_runtime_probe
      core/tile/tests/hip_runtime_probe.cpp)
    target_link_libraries(tides_hip_runtime_probe PRIVATE tides_hip_tile)
    set_target_properties(tides_hip_runtime_probe PROPERTIES
      LANGUAGE HIP
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    add_test(NAME hip_runtime_probe COMMAND tides_hip_runtime_probe)
    set_tests_properties(hip_runtime_probe PROPERTIES
      SKIP_RETURN_CODE 77)

    # HIP test: GEMM correctness.
    add_executable(tides_hip_gemm_tests
      core/tile/tests/hip_gemm_tests.cpp)
    target_link_libraries(tides_hip_gemm_tests PRIVATE tides_hip_tile)
    set_target_properties(tides_hip_gemm_tests PROPERTIES
      LANGUAGE HIP
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    add_test(NAME hip_grouped_gemm_fp64_oracle COMMAND tides_hip_gemm_tests)
    set_tests_properties(hip_grouped_gemm_fp64_oracle PROPERTIES
      SKIP_RETURN_CODE 77)

    # HIP test: Ozaki f64e.
    add_executable(tides_hip_ozaki_tests
      core/tile/tests/hip_ozaki_tests.cpp)
    target_link_libraries(tides_hip_ozaki_tests PRIVATE tides_hip_tile)
    set_target_properties(tides_hip_ozaki_tests PROPERTIES
      LANGUAGE HIP
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    add_test(NAME hip_ozaki_f64e_gemm COMMAND tides_hip_ozaki_tests)
    set_tests_properties(hip_ozaki_f64e_gemm PROPERTIES
      SKIP_RETURN_CODE 77)

    # HIP test: SpGEMM.
    add_executable(tides_hip_spgemm_tests
      core/tile/tests/hip_spgemm_tests.cpp)
    target_link_libraries(tides_hip_spgemm_tests PRIVATE tides_hip_tile)
    set_target_properties(tides_hip_spgemm_tests PROPERTIES
      LANGUAGE HIP
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    add_test(NAME hip_filtered_spgemm COMMAND tides_hip_spgemm_tests)
    set_tests_properties(hip_filtered_spgemm PROPERTIES
      SKIP_RETURN_CODE 77)
  endif()
endif()
