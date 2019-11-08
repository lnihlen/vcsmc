add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/catchuppixels.c"
    COMMAND "m4"
    ARGS "${CMAKE_CURRENT_SOURCE_DIR}/src/z26/catchuppixels.m4" > "${CMAKE_CURRENT_BINARY_DIR}/catchuppixels.c"
    VERBATIM  # prevent cmake from helpfully escaping the stdio redirection >
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cpu.c"
    COMMAND "m4"
    ARGS "${CMAKE_CURRENT_SOURCE_DIR}/src/z26/cpu.m4" > "${CMAKE_CURRENT_BINARY_DIR}/cpu.c"
    VERBATIM
)

add_library(libz26 STATIC
    include/libz26/libz26.h
    src/libz26.c
)

add_dependencies(libz26
    "${CMAKE_CURRENT_BINARY_DIR}/catchuppixels.c"
    "${CMAKE_CURRENT_BINARY_DIR}/cpu.c"
)

target_include_directories(libz26 PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

