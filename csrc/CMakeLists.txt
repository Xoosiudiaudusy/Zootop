cmake_minimum_required(VERSION 3.18)
project(negpluribus_fastcore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_fastcore bindings.cpp)
target_include_directories(_fastcore PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# The GPU trainer (gpucfr.cu) when a CUDA compiler is found (CUDA Toolkit 12.8+ for sm_120 = RTX 50xx):
# a static library with its own flags, linked into the module.  -DNEGP_CUDA=OFF skips it.
option(NEGP_CUDA "build the GPU trainer when a CUDA compiler is found" ON)
if(NEGP_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "89-real;120")  # RTX 40xx (SASS), RTX 50xx (SASS + PTX)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(negp_gpu STATIC gpucfr.cu)
    set_target_properties(negp_gpu PROPERTIES CUDA_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
    target_include_directories(negp_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(negp_gpu PRIVATE NEGP_WITH_CUDA=1)
    # no fused multiply-add, IEEE division and square root: the doubles of the CPU trainers
    target_compile_options(negp_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -prec-div=true -prec-sqrt=true -O3>)
    if(MSVC)
      # CCCL (CUB) of CUDA 13 refuses MSVC's traditional preprocessor: the host compiler needs the conforming one
      target_compile_options(negp_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    endif()
    target_link_libraries(negp_gpu PUBLIC CUDA::cudart_static)
    target_link_libraries(_fastcore PRIVATE negp_gpu)
    target_compile_definitions(_fastcore PRIVATE NEGP_WITH_CUDA=1)
    message(STATUS "negpluribus: GPU trainer ON (${CMAKE_CUDA_COMPILER}, archs ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "negpluribus: no CUDA compiler found, GPU trainer OFF")
  endif()
endif()

if(MSVC)
  # /fp:precise keeps IEEE semantics; contraction into FMA is disabled in the sources with
  # #pragma fp_contract so the single-thread trainer matches the Python reference bit for bit.
  target_compile_options(_fastcore PRIVATE /O2 /fp:precise /EHsc /W3 /wd4244 /wd4267)
else()
  target_compile_options(_fastcore PRIVATE -O3 -ffp-contract=off -Wall)
endif()

# drop the extension straight into the package so `import negpluribus._fastcore` works
set(NEGP_PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../negpluribus")
set_target_properties(_fastcore PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${NEGP_PKG_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${NEGP_PKG_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${NEGP_PKG_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NEGP_PKG_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${NEGP_PKG_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${NEGP_PKG_DIR}"
)
