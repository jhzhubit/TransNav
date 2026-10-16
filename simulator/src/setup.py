from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Get the project root directory containing setup.py.
project_root = os.path.dirname(os.path.abspath(__file__))
# Locate the header directory relative to the project root.
include_dir = os.path.join(project_root, "include")  # Use an absolute path for reliable header lookup.

# 1. Define the CUDA extension and its header search directory.
ext_modules = [
    CUDAExtension(
        name="sensor_simulator_cuda",  # Extension module name used when importing.
        sources=[
            "src/sensor_simulator.cpp",  # C++ source file.
            "src/sensor_simulator_kernel.cu"  # CUDA source file.
        ],
        # Specify the directory where the compiler searches for headers.
        include_dirs=[include_dir]
    )
]

# 2. Configure package metadata and optional header installation.
setup(
    name="sensor_simulator",  # Package name.
    version="0.1.0",
    packages=find_packages(),  # Discover pure Python packages automatically.
    ext_modules=ext_modules,  # Register the extension modules.
    cmdclass={"build_ext": BuildExtension},  # Build extensions with PyTorch's BuildExtension.
    # Optionally install headers with the package for use by other modules.
    # package_data={
    #     # Format: {package_name: [header_paths]}.
    #     "sensor_simulator": [  # Use the package name if the headers are inside a Python package.
    #         "include/*.h",    # Match all .h files in the include directory.
    #         "include/*.cuh"   # Match CUDA .cuh headers.
    #     ]
    # }
)
