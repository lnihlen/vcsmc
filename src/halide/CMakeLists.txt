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
    list(APPEND vcsmc_halide_libs "${GEN_NAME}")
endforeach()

