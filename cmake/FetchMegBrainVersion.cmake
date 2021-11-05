# Parses the version set in src/core/include/megbrain/version.h
# Exports the following variables:
# MGB_VER_MAJOR: Major version
# MGB_VER_MINOR: Minor version
# MGB_VER_PATCH: Patch version
# MGB_IS_DEV: Is development version
# MGB_VER_STRING: Version string
option(MGB_FORCE_DEV_VERSION "Force -dev tag in version stamp" OFF)

file (READ "${CMAKE_CURRENT_SOURCE_DIR}/src/core/include/megbrain/version.h" content)

string (REGEX MATCH "MGB_MAJOR +([0-9]+)" _ ${content})
set (MGB_VER_MAJOR ${CMAKE_MATCH_1})

string (REGEX MATCH "MGB_MINOR +([0-9]+)" _ ${content})
set (MGB_VER_MINOR ${CMAKE_MATCH_1})

string (REGEX MATCH "MGB_PATCH *([0-9]+)" _ ${content})
set (MGB_VER_PATCH ${CMAKE_MATCH_1})

string (REGEX MATCH "MGE_MAJOR +([0-9]+)" _ ${content})
set (MGE_VER_MAJOR ${CMAKE_MATCH_1})

string (REGEX MATCH "MGE_MINOR +([0-9]+)" _ ${content})
set (MGE_VER_MINOR ${CMAKE_MATCH_1})

string (REGEX MATCH "MGE_PATCH *([0-9]+)" _ ${content})
set (MGE_VER_PATCH ${CMAKE_MATCH_1})

string (REGEX MATCH "MGE_EXTRA_NAME *\"(.*)\"" _ ${content})
set (MGE_EXTRA_NAME ${CMAKE_MATCH_1})

if (MGB_FORCE_DEV_VERSION)
    set (MGB_IS_DEV 1)
else()
    string (REGEX MATCH "MGB_IS_DEV +([01])" _ ${content})
    set (MGB_IS_DEV ${CMAKE_MATCH_1})
endif()

if (DEFINED MGB_VER_MAJOR)
    set (MGB_VER_STRING "${MGB_VER_MAJOR}.${MGB_VER_MINOR}.${MGB_VER_PATCH}")
else()
    set (MGB_VER_STRING "${MGE_VER_MAJOR}.${MGE_VER_MINOR}.${MGE_VER_PATCH}")
endif(DEFINED MGB_VER_MAJOR)
if (MGB_IS_DEV)
    set (MGB_VER_STRING "${MGB_VER_STRING}-dev")
endif()

message(STATUS "Building MegBrain ${MGB_VER_STRING}")
