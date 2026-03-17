cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(deep_cfr)

# Fix for newer CUDA architectures not known to older libtorch
# RTX 5090 = compute 12.0, but libtorch 2.2 doesn't know it
# Use 8.9 (Ada) as compatible fallback, or set to native
set(TORCH_CUDA_ARCH_LIST "8.9")
set(ENV{TORCH_CUDA_ARCH_LIST} "8.9")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(deep_cfr deep_cfr.cpp)
target_link_libraries(deep_cfr "${TORCH_LIBRARIES}")
set_property(TARGET deep_cfr PROPERTY CXX_STANDARD 17)

# Copy torch shared libs on Windows
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET deep_cfr POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:deep_cfr>)
endif()
