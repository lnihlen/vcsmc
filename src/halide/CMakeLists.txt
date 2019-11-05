## see https://github.com/halide/Halide/blob/master/README_cmake.md
## need to set(HALIDE_DISTRIB_DIR) in some smart way
include("${HALIDE_DISTRIB_DIR}/halide.cmake")

add_library(vcsmc_halide_lib)

set(vcsmc_halide_generators
    ciede_2k_generator.cc
    covariance_generator.cc
    mean_generator.cc
    rgb_to_lab_generator.cc
    ssim_generator.cc
    variance_generator.cc
)

foreach(GEN_SRC ${vcsmc_halide_generators})
    string(REPLACE "_generator.cc" "" GEN_NAME ${GEN_SRC})
    halide_library("${GEN_NAME}" SRCS ${GEN_SRC})
    target_link_libraries(vcsmc_halide_lib PUBLIC "${GEN_NAME}")
endforeach()

