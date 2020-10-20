# Parses the version set in src/core/include/megbrain/version.h
# Exports the following variables:
# MGB_VER_MAJOR: Major version
# MGB_VER_MINOR: Minor version
# MGB_VER_PATCH: Patch version
# MGB_IS_DEV: Is development version
# MGB_VER_STRING: Version string
option(MGB_FORCE_DEV_VERSION "Force -dev tag in version stamp" OFF)

file (READ "${CMAKE_SOURCE_DIR}/src/core/include/megbrain/version.h" content)

string (REGEX MATCH "MGB_MAJOR +([0-9]+)" _ ${content})
set (MGB_VER_MAJOR ${CMAKE_MATCH_1})

string (REGEX MATCH "MGB_MINOR +([0-9]+)" _ ${content})
set (MGB_VER_MINOR ${CMAKE_MATCH_1})

string (REGEX MATCH "MGB_PATCH *([0-9]+)" _ ${content})
set (MGB_VER_PATCH ${CMAKE_MATCH_1})

if (MGB_FORCE_DEV_VERSION)
    set (MGB_IS_DEV 1)
else()
    string (REGEX MATCH "MGB_IS_DEV +([01])" _ ${content})
    set (MGB_IS_DEV ${CMAKE_MATCH_1})
endif()

set (MGB_VER_STRING "${MGB_VER_MAJOR}.${MGB_VER_MINOR}.${MGB_VER_PATCH}")
if (MGB_IS_DEV)    
    set (MGB_VER_STRING "${MGB_VER_STRING}-dev")
endif()

message(STATUS "Building MegBrain ${MGB_VER_STRING}")
