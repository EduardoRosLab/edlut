################################################################################
################# - MAKEFILE STATIC VARIABLES - ################################
################################################################################

exe-sources   := ${sources} ${exe-source-file}
mex-sources   := ${sources} ${mex-source-file}

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

mextarget		 := $(mexdir)/$(packagename).$(mexsuffix)

automakefile := make.auto
