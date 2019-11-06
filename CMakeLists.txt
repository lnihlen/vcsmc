# Allow specifying project version.
cmake_policy(SET CMP0048 NEW)

project(vcsmc
        VERSION 0.0.1)

cmake_minimum_required(VERSION 3.9)

add_subdirectory(third_party)
add_subdirectory(src)

