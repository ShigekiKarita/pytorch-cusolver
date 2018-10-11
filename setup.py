from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

conda = os.getenv("CONDA_PREFIX")
if conda:
    inc = [conda + "/include"]
else:
    inc = []

libname = "torch_cusolver"
setup(name=libname,
      ext_modules=[CppExtension(
          libname,
          [libname + '.cpp'],
          include_dirs=inc,
          libraries=["cusolver", "cublas"],
          extra_compile_args={'cxx': ['-g', '-DDEBUG'],
                              'nvcc': ['-O2']}
          # extra_compile_args=["-fPIC"]
      )],
      cmdclass={'build_ext': BuildExtension})
