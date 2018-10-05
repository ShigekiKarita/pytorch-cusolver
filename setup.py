from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

libname = "torch_cusolver"
setup(name=libname,
      ext_modules=[CppExtension(
          libname,
          [libname + '.cpp'],
          libraries=["cusolver"],
          extra_compile_args={'cxx': ['-g', '-DDEBUG'],
                              'nvcc': ['-O2']}
          # extra_compile_args=["-fPIC"]
      )],
      cmdclass={'build_ext': BuildExtension})
