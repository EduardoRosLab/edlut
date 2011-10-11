################################################################################
################# - MAKEFILE STATIC VARIABLES - ################################
################################################################################

exe-sources   := ${sources} ${exe-source-file}
mex-sources   := ${sources} ${mex-source-file}
sfunction-sources := ${sources} ${sfunction-source-file}

objects       := $(filter %.o,$(subst   .c,.o,$(sources)))
objects       += $(filter %.o,$(subst  .cc,.o,$(sources)))
objects       += $(filter %.o,$(subst .cpp,.o,$(sources)))
dependencies  := $(subst .o,.d,$(objects))

exe-objects       := $(filter %.o,$(subst   .c,.o,$(exe-sources)))
exe-objects       += $(filter %.o,$(subst  .cc,.o,$(exe-sources)))
exe-objects       += $(filter %.o,$(subst .cpp,.o,$(exe-sources)))
exe-dependencies  := $(subst .o,.d,$(exe-objects))

mex-objects       := $(filter %.o,$(subst   .c,.o,$(mex-sources)))
mex-objects       += $(filter %.o,$(subst  .cc,.o,$(mex-sources)))
mex-objects       += $(filter %.o,$(subst .cpp,.o,$(mex-sources)))
mex-dependencies  := $(subst .o,.d,$(mex-objects))

sfunction-objects       := $(filter %.o,$(subst   .c,.o,$(sfunction-sources)))
sfunction-objects       += $(filter %.o,$(subst  .cc,.o,$(sfunction-sources)))
sfunction-objects       += $(filter %.o,$(subst .cpp,.o,$(sfunction-sources)))
sfunction-dependencies  := $(subst .o,.d,$(sfunction-objects))

ifeq ($(parallelize),true)
# parallel compilation variables
   libtarget     := $(libdir)/lib$(packagename).a
   exetarget     := $(bindir)/$(packagename)
   pkgconfigfile := $(packagename).pc
else
   libtarget     := $(libdir)/lib$(packagename).a
   exetarget     := $(bindir)/$(packagename)
   pkgconfigfile := $(packagename).pc
endif

ARCH 		:= $(shell getconf LONG_BIT)

mexsuffix 	:= $(mex$(ARCH)suffix)

mextarget	:= $(mexdir)/$(packagename).$(mexsuffix)

sfunctiontarget	 := $(sfunctiondir)/$(packagename).$(mexsuffix)

automakefile := make.auto
