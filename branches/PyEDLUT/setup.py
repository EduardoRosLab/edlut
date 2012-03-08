from distutils.core import setup, Extension

edlutmodule = Extension('edlut',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0 rev 1')],
                    include_dirs = ['include'],
                    libraries = ['EDLUTKernel'],
                    library_dirs = ['lib'],
                    sources = ['src/interface/python/edlutmodule.cpp'])

setup (name = 'edlut',
       version = '1.0rev1',
       description = 'EDLUT for Python',
       author = 'Jesus A. Garrido',
       author_email = 'jgarrido@atc.ugr.es',
       url = 'http://edlut.googlecode.com',
       long_description = '''EDLUT interface for Python.''',
       ext_modules = [edlutmodule])
