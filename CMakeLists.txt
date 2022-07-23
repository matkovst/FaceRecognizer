cmake_minimum_required(VERSION 3.15)
project(FaceRecognizer)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
if(NOT OpenCV_DIR)
    message(FATAL_ERROR "You must specify OpenCV_DIR")
endif()
find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR} COMPONENTS core imgproc videoio highgui dnn)

# Torch
if (NOT Torch_DIR)
    message(FATAL_ERROR "You must specify Torch_DIR")
endif()
find_package(Torch HINTS ${Torch_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(NOT TORCH_FOUND)
    message(FATAL_ERROR "Could not find LibTorch")
endif()
include_directories( ${TORCH_INCLUDE_DIRS} )
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/src/*.h*)
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

MACRO(add_program NAME MAIN_FILE)
    add_executable(
        ${NAME} ${MAIN_FILE} ${HEADERS} ${SOURCES})

    target_include_directories(
        ${NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

    target_link_libraries(
        ${NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

    # The following code block is suggested to be used on Windows.
    # According to https://github.com/pytorch/pytorch/issues/25457,
    # the DLLs need to be copied to avoid memory errors.
    if (MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET ${NAME}
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:${NAME}>)
    endif (MSVC)

    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "Install prefix" FORCE)
    set(INSTALL_BIN_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "Path where exe and dll will be installed")
    install(
        TARGETS ${NAME}
        CONFIGURATIONS Release
        RUNTIME DESTINATION bin
        )
ENDMACRO()

add_program(FaceRecognizer main_facerecognizer.cpp)
add_program(FaceCollector main_facecollector.cpp)