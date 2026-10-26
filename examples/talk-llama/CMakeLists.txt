if (WHISPER_SDL2)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # --- llama.cpp version ---
    # WHY: нужно до configure_file(llama-version.h.in).
    set(LLAMA_VERSION_MAJOR 0)
    set(LLAMA_VERSION_MINOR 7)
    set(LLAMA_VERSION_PATCH 0)
    set(LLAMA_VERSION_BASE "${LLAMA_VERSION_MAJOR}.${LLAMA_VERSION_MINOR}.${LLAMA_VERSION_PATCH}")
    set(LLAMA_VERSION "${LLAMA_VERSION_BASE}-dev")
    set(LLAMA_BUILD_COMMIT "unknown")

    # WHY: llama.cpp включает "llama-version.h" (генерируетс€ из .in).
    #      Ѕез этого файла падают llama.cpp, llama-model.cpp, llama-context.cpp,
    #      llama-vocab.cpp и ещЄ ~15 файлов.
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/llama-version.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h
        @ONLY
    )

    # WHY: file(GLOB ...) только дл€ моделей. ƒл€ llama-kv-cache-* и
    #      llama-memory-* используем €вный список (иначе дубликаты).
    file(GLOB SRC_MODELS models/*.cpp)

    set(TARGET whisper-talk-llama)
    add_executable(${TARGET}
        talk-llama.cpp

        # --- llama.cpp core ---
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

        # --- KV cache variants ---
        llama-kv-cache.cpp
        llama-kv-cache-iswa.cpp
        llama-kv-cache-dsa.cpp
        llama-kv-cache-dsa-iswa.cpp
        llama-kv-cache-msa.cpp
        llama-kv-cache-dsv4.cpp

        # --- Memory variants ---
        llama-memory.cpp
        llama-memory-recurrent.cpp
        llama-memory-hybrid.cpp
        llama-memory-hybrid-iswa.cpp
        llama-memory-hybrid-idx.cpp

        # --- MoE cache ---
        llama-moe-cache.cpp

        # --- Model I/O ---
        llama-mmap.cpp
        llama-model-loader.cpp
        llama-model-saver.cpp
        llama-model.cpp
        llama-quant.cpp
        llama-sampler.cpp
        llama-vocab.cpp

        # --- Unicode ---
        unicode.cpp
        unicode-data.cpp

        # --- Model architectures (GLOB) ---
        ${SRC_MODELS}
    )

    # WHY: ${CMAKE_CURRENT_BINARY_DIR} об€зателен Ч там лежит
    #      сгенерированный llama-version.h.
    target_include_directories(${TARGET} PRIVATE
        .
        ${CMAKE_CURRENT_BINARY_DIR}
        ${SDL2_INCLUDE_DIRS}
    )

    find_package(CURL REQUIRED)
    target_link_libraries(${TARGET} PRIVATE CURL::libcurl)

    # WHY: LLAMA_VERSION приходит из llama-version.h Ч не дублируем
    #      через target_compile_definitions.

    target_link_libraries(${TARGET} PRIVATE
        common
        common-sdl
        whisper
        ${SDL2_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    install(TARGETS ${TARGET} RUNTIME)

    if(WIN32)
        target_compile_definitions(${TARGET} PRIVATE _WIN32_WINNT=0x0602)
        target_compile_definitions(${TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()

    if(MSVC)
        # WHY: llama-quant.cpp, llama-graph.cpp Ч огромные файлы,
        #      без /bigobj MSVC падает на C1128.
        target_compile_options(${TARGET} PRIVATE /bigobj)
         target_compile_options(${TARGET} PRIVATE /bigobj /utf-8)
    endif()

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