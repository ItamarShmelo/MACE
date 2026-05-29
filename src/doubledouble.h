#ifndef COMPTON_DOUBLEDOUBLE_ROOT_SHIM_H
#define COMPTON_DOUBLEDOUBLE_ROOT_SHIM_H

#if __has_include("../build/_deps/doubledouble-src/include/doubledouble.h")
#include "../build/_deps/doubledouble-src/include/doubledouble.h"
#elif __has_include(<doubledouble.h>)
#include <doubledouble.h>
#else
#error "Unable to locate doubledouble.h; run CMake configure to fetch dependencies."
#endif

#endif
