cmake_minimum_required(VERSION 3.14)
project(voxbridge_engines LANGUAGES C CXX)

# Builds ONE self-contained "VoxBridge engine" DLL for a specific CPU ISA target (or
# Vulkan): either a transcription engine (whisper.cpp/ggml) or a small embedded-LLM
# engine (llama.cpp/ggml), controlled by VOXBRIDGE_TARGET, statically linked together
# with a thin C ABI shim. Each variant is its own out-of-source build directory against
# this same CMakeLists.txt/vendored-project tree, configured with different -D flags -
# e.g.:
#
#   cmake -B build-baseline -DVOXBRIDGE_TARGET=whisper -DVOXBRIDGE_VARIANT=cpu-baseline
#   cmake -B build-avx2     -DVOXBRIDGE_TARGET=whisper -DVOXBRIDGE_VARIANT=cpu-avx2 -DGGML_AVX2=ON
#   cmake -B build-vulkan   -DVOXBRIDGE_TARGET=whisper -DVOXBRIDGE_VARIANT=vulkan    -DGGML_VULKAN=ON
#   cmake -B build-llm-cpu  -DVOXBRIDGE_TARGET=llm      -DVOXBRIDGE_VARIANT=cpu-baseline
#   cmake -B build-llm-vk   -DVOXBRIDGE_TARGET=llm      -DVOXBRIDGE_VARIANT=vulkan    -DGGML_VULKAN=ON
#
# Because each variant is a fully separate CMake configure+build producing
# its own standalone DLL, there's no symbol collision between variants -
# unlike GGML_BACKEND_DL, which tries to make ggml itself produce multiple
# ISA variants of the SAME symbol names as sibling dlopen-able modules
# within one build. The consuming application therefore builds standalone variants.
# whisper.cpp and llama.cpp each vendor their own ggml, but since a whisper engine DLL
# and an llm engine DLL are never built in the same configure run (and only one variant
# of either family is ever dlopen'd into the process at once), there's no collision
# between the two families either.
#
# This directory (voxbridge/) is intentionally self-contained: no path outside
# voxbridge/ is referenced anywhere in here, so it can be lifted into its own
# repository later without any rework.

set(VOXBRIDGE_TARGET "whisper" CACHE STRING "Engine family to build: whisper or llm")
set(VOXBRIDGE_VARIANT "cpu-baseline" CACHE STRING "Engine variant name, e.g. cpu-baseline, cpu-avx2, vulkan")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# Required on Linux/Unix: these static libs get linked into our SHARED voxbridge_engine
# library below, and a static lib linked into a shared object must be position-
# independent code. Not needed on Windows (MSVC doesn't have this restriction), but
# harmless there.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE) # never bake in the build machine's exact CPU

# Vendored whisper.cpp/llama.cpp/parakeet.cpp/ggml is upstream code we don't touch -
# silence its benign numeric-narrowing warnings (int64_t/double -> int/float) so CI logs
# aren't flooded with thousands of lines that aren't actionable here.
if(MSVC)
  add_compile_options(/wd4244 /wd4267)
else()
  add_compile_options(-Wno-conversion -Wno-narrowing)
endif()

if(VOXBRIDGE_TARGET STREQUAL "whisper")
  set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)

  add_subdirectory(whisper.cpp)

  add_library(voxbridge_engine SHARED
      shim/voxbridge_engine.cpp
  )
  target_include_directories(voxbridge_engine PRIVATE whisper.cpp/include)
  target_link_libraries(voxbridge_engine PRIVATE whisper)
  target_compile_definitions(voxbridge_engine PRIVATE
      VOXBRIDGE_ENGINE_VARIANT_NAME="${VOXBRIDGE_VARIANT}"
  )

elseif(VOXBRIDGE_TARGET STREQUAL "llm")
  set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)

  add_subdirectory(llama.cpp)

  add_library(voxbridge_engine SHARED
      shim/voxbridge_llm.cpp
  )
  target_include_directories(voxbridge_engine PRIVATE llama.cpp/include llama.cpp/ggml/include)
  target_link_libraries(voxbridge_engine PRIVATE llama)
  target_compile_definitions(voxbridge_engine PRIVATE
      VOXBRIDGE_LLM_VARIANT_NAME="${VOXBRIDGE_VARIANT}"
  )

else()
  message(FATAL_ERROR "Unknown VOXBRIDGE_TARGET '${VOXBRIDGE_TARGET}' - expected 'whisper' or 'llm'")
endif()

# Keep the existing "voxbridge_engine_<variant>" naming for whisper (already what the
# Rust loader and build-engines.mjs expect, and what's already deployed) - only the new
# llm family gets a distinct "voxbridge_llm_<variant>" name, so both can be told apart
# on disk and never confused for each other.
if(VOXBRIDGE_TARGET STREQUAL "whisper")
  set(VOXBRIDGE_OUTPUT_NAME "voxbridge_engine_${VOXBRIDGE_VARIANT}")
else()
  set(VOXBRIDGE_OUTPUT_NAME "voxbridge_llm_${VOXBRIDGE_VARIANT}")
endif()

set_target_properties(voxbridge_engine PROPERTIES
    OUTPUT_NAME "${VOXBRIDGE_OUTPUT_NAME}"
    # CMake prefixes shared library output with "lib" on Linux/macOS by default (no
    # prefix on Windows) - force no prefix everywhere so the built filename matches
    # exactly what the Rust loader and build-engines.mjs already expect on every
    # platform, rather than teaching them about a per-platform prefix.
    PREFIX ""
)
