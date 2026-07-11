// CPU-only compilation of xc_engine (when CUDA is not available).
// The .cu file has #ifdef TIDES_HAVE_CUDA guards, so it compiles as C++.
#include "grid/xc/xc_engine.cu"
