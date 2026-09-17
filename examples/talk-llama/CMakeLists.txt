if (WHISPER_SDL2)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # WHY: Use GLOB to automatically pick up new files matching the pattern.
    #      Note: CMake does not detect newly added files until re-configure.
    file(GLOB SRC_KV_CACHE llama-kv-cache-*.cpp)
    file(GLOB SRC_MEMORY llama-memory-*.cpp)
    file(GLOB SRC_MODELS models/*.cpp)

    set(TARGET whisper-talk-llama)
    add_executable(${TARGET} talk-llama.cpp
        llama.cpp
        llama-adapter.cpp
        llama-arch.cpp
        llama-batch.cpp
        llama-chat.cpp
        llama-context.cpp
        llama-cparams.cpp
        llama-grammar.cpp
        llama-graph.cpp
        llama-hparams.cpp
        llama-impl.cpp
        llama-io.cpp
        llama-kv-cache.cpp
        llama-kv-cache-iswa.cpp
        llama-kv-cache-dsa.cpp
        llama-kv-cache-dsv4.cpp
        llama-memory-recurrent.cpp
        llama-memory-hybrid.cpp
        llama-memory-hybrid-iswa.cpp
        ${SRC_KV_CACHE}
        llama-memory.cpp
        ${SRC_MEMORY}
        llama-mmap.cpp
        llama-model-loader.cpp
        llama-model-saver.cpp
        llama-model.cpp
        llama-quant.cpp
        llama-sampler.cpp
        llama-vocab.cpp
        unicode.cpp
        unicode-data.cpp
        ${SRC_MODELS})

    target_include_directories(${TARGET} PRIVATE . ${SDL2_INCLUDE_DIRS})

    # WHY: vcpkg is connected via CMakePresets.json (CMAKE_TOOLCHAIN_FILE).
    #      find_package(CURL) will resolve include and library paths
    #      automatically through the vcpkg toolchain. Do not hard-code
    #      vcpkg paths here — they break portability.
    find_package(CURL REQUIRED)
    target_link_libraries(${TARGET} PRIVATE CURL::libcurl)

    # WHY: -D must NOT be written explicitly in target_compile_definitions.
    #      CMake adds it automatically. LLAMA_VERSION is a placeholder.
    target_compile_definitions(${TARGET} PRIVATE LLAMA_VERSION="0.0.0")

    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${SDL2_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TARGET} RUNTIME)

    if(WIN32)
        # WHY: _WIN32_WINNT must NOT be prefixed with -D.
        #      0x0602 = Windows 8. It requires Windows 8.1 or later
        #      for PrefetchVirtualMemory. Use 0x0A00 for Windows 10+.
        target_compile_definitions(${TARGET} PRIVATE _WIN32_WINNT=0x0602)
    endif()

    # WHY: CMake links SDL2.lib but does NOT copy SDL2.dll.
    #      The DLL must sit next to the .exe for the app to start.
    #      Copy it to build/bin/Release/ (or Debug/) automatically.
    if(WIN32)
        set(SDL2_DLL_PATH "${CMAKE_SOURCE_DIR}/SDL2/lib/x64/SDL2.dll")

        if(EXISTS "${SDL2_DLL_PATH}")
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${SDL2_DLL_PATH}"
                    "$<TARGET_FILE_DIR:${TARGET}>"
                COMMENT "[SDL2] Copied SDL2.dll to $<TARGET_FILE_DIR:${TARGET}>"
            )
            message(STATUS "[SDL2] DLL found: ${SDL2_DLL_PATH}")
            message(STATUS "[SDL2] Will be copied to: $<TARGET_FILE_DIR:${TARGET}>")
        else()
            message(WARNING "[SDL2] SDL2.dll not found at ${SDL2_DLL_PATH}")
        endif()
    endif()

    include(DefaultTargetOptions)
endif()