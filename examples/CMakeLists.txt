# dependencies

find_package(Threads REQUIRED)

# third-party

if (WHISPER_SDL2)
    # SDL2
    find_package(SDL2 REQUIRED)

    string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

    message(STATUS "SDL2_INCLUDE_DIRS = ${SDL2_INCLUDE_DIRS}")
    message(STATUS "SDL2_LIBRARIES    = ${SDL2_LIBRARIES}")
endif()

# common

set(TARGET common)

unset(COMMON_EXTRA_LIBS)

# WHY: WHISPER_COMMON_FFMPEG is OFF by default and ffmpeg-transcode.cpp
#      has been removed from this project. Left commented out in case
#      FFmpeg support is ever re-enabled.
# if (WHISPER_COMMON_FFMPEG)
#     find_package(FFmpeg REQUIRED)
#     if (NOT ${FFMPEG_FOUND})
#         message(FATAL_ERROR "Cannot find ffmpeg libs/headers")
#     endif()
#     include_directories(${FFMPEG_INCLUDE_DIRS})
#     add_compile_definitions(WHISPER_COMMON_FFMPEG)
#     list(APPEND COMMON_EXTRA_LIBS ${FFMPEG_LIBRARIES})
#     set(COMMON_SOURCES_FFMPEG ffmpeg-transcode.cpp)
# endif()

add_library(${TARGET} STATIC
    common.h
    common.cpp
    common-ggml.h
    common-ggml.cpp
    common-whisper.h
    common-whisper.cpp
    grammar-parser.h
    grammar-parser.cpp
    )

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE whisper ${COMMON_EXTRA_LIBS} ${CMAKE_DL_LIBS})

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${TARGET} PROPERTIES FOLDER "libs")

if (WHISPER_SDL2)
    # common-sdl

    set(TARGET common-sdl)

    add_library(${TARGET} STATIC
        common-sdl.h
        common-sdl.cpp
        )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PUBLIC  ${SDL2_INCLUDE_DIRS})
    target_link_libraries     (${TARGET} PRIVATE ${SDL2_LIBRARIES})

    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

# add json lib
add_library(json_cpp INTERFACE)
target_include_directories(json_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# WHY: Only talk-llama is built in this project. The whisper.wasm,
#      command.wasm and addon.node subdirectories do not exist here,
#      so the EMSCRIPTEN / CMAKE_JS_VERSION branches are removed to
#      avoid accidental "directory not found" errors.
if (WHISPER_SDL2)
    add_subdirectory(talk-llama)
    if (GGML_SYCL)
        add_subdirectory(sycl)
    endif()
endif()