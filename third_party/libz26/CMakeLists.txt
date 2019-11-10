add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/m4/catchuppixels.c" "${CMAKE_CURRENT_BINARY_DIR}/m4/cpu.c"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/m4"
    COMMAND m4 "${CMAKE_CURRENT_SOURCE_DIR}/src/z26/catchuppixels.m4" > "${CMAKE_CURRENT_BINARY_DIR}/m4/catchuppixels.c"
    COMMAND m4 "${CMAKE_CURRENT_SOURCE_DIR}/src/z26/cpu.m4" > "${CMAKE_CURRENT_BINARY_DIR}/m4/cpu.c"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/z26/catchuppixels.m4" "${CMAKE_CURRENT_SOURCE_DIR}/src/z26/cpu.m4"
    VERBATIM
)

add_custom_target(libz26_m4
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/m4/catchuppixels.c" "${CMAKE_CURRENT_BINARY_DIR}/m4/cpu.c"
)

add_library(libz26 STATIC
    include/libz26/libz26.h
    src/libz26.c
)

add_dependencies(libz26
    libz26_m4
)

target_include_directories(libz26 PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

