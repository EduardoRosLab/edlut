--------------------------------
-     How-to compile PyEDLUT   -
--------------------------------

1. Compile EDLUT as a library. From the EDLUT source root directory type in the command line:

> make library

If everything goes well, a new file (libEDLUTKernel.a) should be created in the lib folder.

2. Build the python interface with EDLUT:

> python setup.py build

In the folder build/lib.linux-<systemid> a new dynamic library has been built. If you want to use EDLUT from Python, you will have to indicate that folder in your Python path.

3. However, you could also install EDLUT in your Python libraries. In order to do that, just type:

> python setup.py install

It might require superuser permissions.

4. Check EDLUT has been correctly installed. Go to example dir and run test.py.

> cd example
> python test.py

Enjoy your EDLUT :)

For any comments, suggestions or bugs, email to Jesus Garrido (jgarrido@atc.ugr.es).