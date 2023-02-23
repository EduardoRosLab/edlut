# `python` folder

This directory contains the source code of PyEDLUT, the Python bindings
to the EDLUT kernel. PyEDLUT will be compiled together with EDLUT by default. If you want to
disable it, pass

    -Dwith-python=OFF

as an argument to `cmake`. By default, `make install` will install
PyEDLUT to `$(pyexecdir)`, which is often expanded as follows:

    $(prefix)/lib{,64}/pythonX.Y/site-packages/edlut


To force the usage of a specific Python version pass

    -Dwith-python=2  or  -Dwith-python=3

as an argument to `cmake`.

To make the PyEDLUT module available to the Python interpreter, add the
PyEDLUT installation path (without the final '/nest') to the PYTHONPATH
environment variable.