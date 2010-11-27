#============================================================================#
# This program is free software; you can redistribute it		     #
# and/or modify it under the terms of the GNU General Public License	     #
# version 2 (or higher) as published by the Free Software Foundation.	     #
# 									     #
# This program is distributed in the hope that it will be useful, but	     #
# WITHOUT ANY WARRANTY; without even the implied warranty of		     #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU	     #
# General Public License for more details.				     #
# 									     #
# Written and (C) by							     #
# Engin Tola								     #
# 									     #
# web   : http://cvlab.epfl.ch/~tola					     #
# email : engin.tola@epfl.ch						     #
# 									     #
#============================================================================#

# for a more detailed explanation visit 
# http://cvlab.epfl.ch/~tola/makefile_heaven.html
MAKEFILE_HEAVEN	:= .

# this is the directory the library will be installed if you issue a
# 'make install' command:
# headers to $(installdir)/$(packagename)/include
# library to $(installdir)/lib
# pkg-file to $(installdir)/lib/pkgconfig/
# 'make install-exe' command:
# executable to $(installdir)/bin/
installdir  := $(HOME)/usr

# this is the name of the package. i.e if it is 'cvlab' the executable
# will be named as 'cvlab' and if this is a library its name will be
# 'libcvlab.a'
packagename := EDLUTKernel

version     := 0.1
author      := $(FULL_USER_NAME)

# you can write a short description here about the package
description := EDLUT Kernel Library

# i'm for gpl but you can edit it yourself
licence     := GPL v3.0 or higher distributed by FSF

# the external libraries and sources are managed (included, linked)
# using the pkg-config program. if you don't have it, you cannot use
# this template to include/link libraries. get it from
# http://pkg-config.freedesktop.org/wiki

# if matlab support -> Matlab paths, includes and flags.
matlabsupport	:= true

# matlab root dir : Folder where Matlab is installed. The root folder
# in order to find the matlab include folder.If you don't need matlab 
# compatibility (mex files), this option won't be used.
matlabrootdir	:= /usr/local/matlabR2009b

# matlab include folder : Path where matlab include files are stored.
# If you don't need matlab compatibility (mex files), this option won't
# be used.
matlabinclude	:= ${matlabrootdir}/extern/include

# matlab mex file suffix : Suffix of generated mex files. E.g. mexglx in Linux, mexw32
# in windows 32 bits, mexw64 in windows 64 bits... See mex documentation for more
# detailed information.
mexsuffix	:= mexglx

# external sources : uses pkg-config as " pkg-config --cflags ${external_sources} "
# if you don't need any source, set it to 'none'
# external_sources := lpp
external_sources := none

# external sources : uses pkg-config as " pkg-config --cflags
# ${external_libraries} " for CXXFLAGS and pkg-config --libs
# ${external_libraries} for library inclusions if you don't need any
# outside library, set it to 'none' the order is important for
# linking. write the name of the package that depends on anothe
# package first.
# external_libraries := kfeature kutility opencv atlas
external_libraries := none

# fortran to c conversion ? I need this for Lapack - ATLAS library
# stuff (also lpp above)
fortran77support := false

# if optimized -> no debug info is produced --> applies -O3 flag if
# set to true
# optimize := true
optimize    := false

# this is for laptops and stuff with intel pentium M processors, if
# you are not sure of your system, just set 'specialize' to false. if
# it is a different one look for the -march option param of gcc and
# write your platforms name optimize for pentium4 ? / disabled if
# optimize is false
specialize  := false
platform    := pentium4

# do you want openmp support ? if you've never heard of it say 'false'
# parallelize := true
parallelize := false

# pthread support
multi-threading := false

# enable sse instruction sets ( sse sse2 )
sse-iset := false

# generate profiler data if true.  
#   ! set the optimize = false if you want annotation support.  
#  !! if you don't compile libraries with this flag, profiler won't be 
#      able to make measurements for those libraries.  
# !!! after running your program, you can see the results with
#      'make gflat' and 'make gcall'
profile := true

# do not change for linux /usr type directory structures. this structure means
# .cpp .cc files reside in ./srcdir and .h files reside in ./include/$(packagename)/
# if you are building a library, the lib$(packagename).a will be in ./lib file.
# libdir      := lib
# srcdir      := src
# includedir  := include
# If you'd like to have everything in the main directory
libdir      	:= ./lib
srcdir      	:= ./src
includedir  	:= ./include
bindir			:= ./bin
mexdir			:= ./mex

# default file to be included when the executable build is required
exe-source-file := ${srcdir}/EDLUTKernel.cpp

# default file to be included when the mex build is required
mex-source-file := ${srcdir}/EDLUTMatlab.cpp

# what to compile ? include .cpp and .c files here in your project
# if you don't have a main() function in one of the sources, you'll get an error
# if you're building an executable. for a library, it won't complain for anything.
communication-sources	:= $(srcdir)/communication/ArrayInputSpikeDriver.cpp \
                           $(srcdir)/communication/ArrayOutputSpikeDriver.cpp \
						   $(srcdir)/communication/CdSocket.cpp \
                           $(srcdir)/communication/ClientSocket.cpp \
                           $(srcdir)/communication/CommunicationDevice.cpp \
                           $(srcdir)/communication/ConnectionException.cpp \
                           $(srcdir)/communication/FileInputSpikeDriver.cpp \
                           $(srcdir)/communication/FileOutputSpikeDriver.cpp \
                           $(srcdir)/communication/FileOutputWeightDriver.cpp \
                           $(srcdir)/communication/InputSpikeDriver.cpp \
                           $(srcdir)/communication/OutputSpikeDriver.cpp \
                           $(srcdir)/communication/OutputWeightDriver.cpp \
                           $(srcdir)/communication/ServerSocket.cpp \
                           $(srcdir)/communication/TCPIPInputOutputSpikeDriver.cpp \
                           $(srcdir)/communication/TCPIPInputSpikeDriver.cpp \
                           $(srcdir)/communication/TCPIPOutputSpikeDriver.cpp \
                           $(srcdir)/communication/Timer.cpp
                           
simulation-sources		:= $(srcdir)/simulation/CommunicationEvent.cpp \
                           $(srcdir)/simulation/EndSimulationEvent.cpp \
                           $(srcdir)/simulation/Event.cpp \
                           $(srcdir)/simulation/EventQueue.cpp \
                           $(srcdir)/simulation/ParameterException.cpp \
                           $(srcdir)/simulation/ParamReader.cpp \
                           $(srcdir)/simulation/SaveWeightsEvent.cpp \
                           $(srcdir)/simulation/Simulation.cpp \
                           $(srcdir)/simulation/Utils.cpp
                           
spike-sources			:= $(srcdir)/spike/AdditiveWeightChange.cpp \
						   $(srcdir)/spike/ActivityRegister.cpp \
                           $(srcdir)/spike/ActivityRegister.cpp \
                           $(srcdir)/spike/EDLUTException.cpp \
                           $(srcdir)/spike/EDLUTFileException.cpp \
                           $(srcdir)/spike/InputSpike.cpp \
                           $(srcdir)/spike/Interconnection.cpp \
                           $(srcdir)/spike/InternalSpike.cpp \
                           $(srcdir)/spike/MultiplicativeWeightChange.cpp \
                           $(srcdir)/spike/Network.cpp \
                           $(srcdir)/spike/Neuron.cpp \
                           $(srcdir)/spike/NeuronModelTable.cpp \
                           $(srcdir)/spike/NeuronType.cpp \
                           $(srcdir)/spike/PropagatedSpike.cpp \
                           $(srcdir)/spike/Spike.cpp \
                           $(srcdir)/spike/WeightChange.cpp \
                           $(srcdir)/spike/SinWeightChange.cpp
                          
sources     := $(communication-sources) \
               $(simulation-sources) \
               $(spike-sources)


################################################################################
####################### LOAD PRESET SETTINGS ###################################
################################################################################

# these are the magic files that this interface depends.

# some temp operations.
include $(MAKEFILE_HEAVEN)/static-variables.makefile

# flag settings for gcc like CXXFLAGS, LDFLAGS...  to see the active
# flag definitions, issue 'make flags' command
include $(MAKEFILE_HEAVEN)/flags.makefile

# rules are defined here. to see a list of the available rules, issue 'make rules'
# command
include $(MAKEFILE_HEAVEN)/rules.makefile
