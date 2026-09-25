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
