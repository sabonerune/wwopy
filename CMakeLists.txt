cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

if(NOT "${SKBUILD}")
  message(
    WARNING
      "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install -r ./requirements.txt
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

if(NOT "${CMAKE_BUILD_TYPE}" AND NOT "${CMAKE_CONFIGURATION_TYPES}")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

find_package(
  Python 3.8 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# WORLD
option(WORLD_BUILD_EXAMPLES "" OFF)
add_subdirectory(ext/World EXCLUDE_FROM_ALL)
# Something is probably wrong.
target_include_directories(
  world
  INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext/World/src>")

# module
nanobind_add_module(wwopy_ext NB_STATIC src/wwopy_ext.cpp)
target_compile_options(wwopy_ext PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
target_link_libraries(wwopy_ext PRIVATE world)
install(TARGETS wwopy_ext LIBRARY DESTINATION wwopy)

# stub file
nanobind_add_stub(
  wwopy_ext_stub
  MODULE wwopy_ext
  OUTPUT wwopy_ext.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:wwopy_ext>
  DEPENDS wwopy_ext
  MARKER_FILE py.typed
  PATTERN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/pattern_file.nb")

install(FILES "${CMAKE_BINARY_DIR}/py.typed" "${CMAKE_BINARY_DIR}/wwopy_ext.pyi"
        DESTINATION wwopy)
if("${SKBUILD_STATE}" STREQUAL "editable")
  # Dev only. Install stub file to source directory.
  install(FILES "${CMAKE_BINARY_DIR}/py.typed"
                "${CMAKE_BINARY_DIR}/wwopy_ext.pyi"
          DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/src/wwopy")
endif()

# Merge licenses
add_custom_command(OUTPUT always_rebuild COMMAND "${CMAKE_COMMAND}" -E echo)
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/LICENSE.txt"
  COMMAND "${Python_EXECUTABLE}" cmake/generate_license.py
          "--dist=${CMAKE_BINARY_DIR}/LICENSE.txt"
  DEPENDS always_rebuild
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Generating ${CMAKE_BINARY_DIR}/LICENSE.txt")
add_custom_target(wwopy_license ALL DEPENDS "${CMAKE_BINARY_DIR}/LICENSE.txt")
install(FILES "${CMAKE_BINARY_DIR}/LICENSE.txt"
        DESTINATION "${SKBUILD_METADATA_DIR}/licenses")
