cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(deep_cfr)

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
