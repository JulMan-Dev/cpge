cmake_minimum_required(VERSION 3.23)

project(cpge)

if ($ENV{CARGO_CFG_TARGET_OS} STREQUAL "macos")
    enable_language(Swift)
    add_library(cpge-native STATIC src/gl/macos.swift)
    target_include_directories(cpge-native PRIVATE src/gl)
endif ()
