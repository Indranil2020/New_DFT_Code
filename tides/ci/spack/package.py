# TIDES Spack recipe (T8.5): one-command install via Spack.
# Usage: spack install tides +cuda arch=linux-ubuntu22.04-x86_64
# Place in a Spack repo or use `spack create` then edit.
#
# This is a recipe template; the actual package.py goes in a Spack repo.
# The recipe documents all dependencies and build options.

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack.package import *


class Tides(CMakePackage):
    """TIDES: TIle-based Democratic Electronic-Structure suite.
    Open-source Kohn-Sham DFT engine spanning 10 to 10^6 atoms on GPU.
    """

    homepage = "https://github.com/tides-dft/tides"
    url = "https://github.com/tides-dft/tides/releases/download/v0.1/tides-0.1.0.tar.gz"
    list_url = "https://github.com/tides-dft/tides/releases"

    version("0.1.0", sha256="TBD")

    variant("cuda", default=True, description="Enable CUDA GPU backend")
    variant("hip", default=False, description="Enable HIP/ROCm GPU backend")
    variant("mpi", default=False, description="Enable MPI multi-node")
    variant("tests", default=True, description="Build test suite")

    depends_on("cmake@3.21:", type="build")
    depends_on("lapack")
    depends_on("blas")

    with when("+cuda"):
        depends_on("cuda@12:")
        conflicts("%gcc@:11", msg="CUDA requires GCC >= 12")

    with when("+hip"):
        depends_on("hip@5:")
        depends_on("rocm-libs")

    with when("+mpi"):
        depends_on("mpi")

    with when("+tests"):
        depends_on("py-pyscf", type="test")
        depends_on("py-h5py", type="test")

    def cmake_args(self):
        args = [
            self.define_from_variant("TIDES_ENABLE_CUDA", "cuda"),
            self.define_from_variant("TIDES_ENABLE_MPI", "mpi"),
            self.define("CMAKE_CXX_STANDARD", "20"),
        ]
        if "+cuda" in self.spec:
            args.append(self.define("CMAKE_CUDA_ARCHITECTURES", "89;90"))
        return args
