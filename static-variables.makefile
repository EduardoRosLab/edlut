################################################################################
################# - MAKEFILE STATIC VARIABLES - ################################
################################################################################

exe-sources   := ${sources} ${exe-source-file}

objects       := $(filter %.o,$(subst   .c,.o,$(sources)))
objects       += $(filter %.o,$(subst  .cc,.o,$(sources)))
objects       += $(filter %.o,$(subst .cpp,.o,$(sources)))
dependencies  := $(subst .o,.d,$(objects))

exe-objects       := $(filter %.o,$(subst   .c,.o,$(exe-sources)))
exe-objects       += $(filter %.o,$(subst  .cc,.o,$(exe-sources)))
exe-objects       += $(filter %.o,$(subst .cpp,.o,$(exe-sources)))
exe-dependencies  := $(subst .o,.d,$(exe-objects))

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

automakefile := make.auto
