# Writes OUTPUT (mace_version.hpp) with MACE_GIT_COMMIT: the git commit of
# SOURCE_DIR, plus "-dirty-<hash>" when src/ has uncommitted or untracked
# changes (the hash covers the diff and the untracked files), so two different
# modified builds never share a cached Compton table.  The file is rewritten
# only when its content changes.  Usage:
#   cmake -DSOURCE_DIR=<repo> -DOUTPUT=<file> -P MaceVersion.cmake
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE result
    ERROR_QUIET)
if(NOT result EQUAL 0)
    set(commit "unknown")
else()
    execute_process(
        COMMAND git diff HEAD -- src
        WORKING_DIRECTORY "${SOURCE_DIR}"
        OUTPUT_VARIABLE diff
        ERROR_QUIET)
    execute_process(
        COMMAND git ls-files --others --exclude-standard -- src
        WORKING_DIRECTORY "${SOURCE_DIR}"
        OUTPUT_VARIABLE untracked
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    set(fingerprint "${diff}")
    if(untracked)
        string(REPLACE "\n" ";" untracked_files "${untracked}")
        foreach(file IN LISTS untracked_files)
            file(READ "${SOURCE_DIR}/${file}" content)
            string(APPEND fingerprint "${file}\n${content}")
        endforeach()
    endif()
    if(fingerprint)
        string(SHA1 hash "${fingerprint}")
        string(SUBSTRING "${hash}" 0 12 hash)
        set(commit "${commit}-dirty-${hash}")
    endif()
endif()
set(content "#pragma once\n#define MACE_GIT_COMMIT \"${commit}\"\n")
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" previous)
endif()
if(NOT "${previous}" STREQUAL "${content}")
    file(WRITE "${OUTPUT}" "${content}")
endif()
