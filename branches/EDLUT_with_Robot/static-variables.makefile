################################################################################
################# - MAKEFILE STATIC VARIABLES - ################################
################################################################################

exe-sources   := ${sources} ${exe-source-file}
rtexe-sources   := ${sources} ${rtexe-source-file}
step-sources   := ${sources} ${step-source-file}
precision-sources   := ${sources} ${prec-source-file}
robot-sources	:= ${sources} ${robot-source-file}
mex-sources   := ${sources} ${mex-source-file}
sfunction-sources := ${sources} ${sfunction-source-file}

objects       := $(filter %.o,$(subst   .c,.o,$(sources)))
objects       += $(filter %.o,$(subst  .cc,.o,$(sources)))
objects       += $(filter %.o,$(subst .cpp,.o,$(sources)))
objects       += $(filter %.o,$(subst .cu,.o,$(sources)))
dependencies  := $(subst .o,.d,$(objects))

exe-objects       := $(filter %.o,$(subst   .c,.o,$(exe-sources)))
exe-objects       += $(filter %.o,$(subst  .cc,.o,$(exe-sources)))
exe-objects       += $(filter %.o,$(subst .cpp,.o,$(exe-sources)))
exe-objects       += $(filter %.o,$(subst .cu,.o,$(exe-sources)))
exe-dependencies  := $(subst .o,.d,$(exe-objects))

rtexe-objects       := $(filter %.o,$(subst   .c,.o,$(rtexe-sources)))
rtexe-objects       += $(filter %.o,$(subst  .cc,.o,$(rtexe-sources)))
rtexe-objects       += $(filter %.o,$(subst .cpp,.o,$(rtexe-sources)))
rtexe-objects       += $(filter %.o,$(subst .cu,.o,$(rtexe-sources)))
rtexe-dependencies  := $(subst .o,.d,$(rtexe-objects))

step-objects       := $(filter %.o,$(subst   .c,.o,$(step-sources)))
step-objects       += $(filter %.o,$(subst  .cc,.o,$(step-sources)))
step-objects       += $(filter %.o,$(subst .cpp,.o,$(step-sources)))
step-objects       += $(filter %.o,$(subst .cu,.o,$(step-sources)))
step-dependencies  := $(subst .o,.d,$(step-objects))

precision-objects       := $(filter %.o,$(subst   .c,.o,$(precision-sources)))
precision-objects       += $(filter %.o,$(subst  .cc,.o,$(precision-sources)))
precision-objects       += $(filter %.o,$(subst .cpp,.o,$(precision-sources)))
precision-objects       += $(filter %.o,$(subst .cu,.o,$(precision-sources)))
precision-dependencies  := $(subst .o,.d,$(precision-objects))

robot-objects       := $(filter %.o,$(subst   .c,.o,$(robot-sources)))
robot-objects       += $(filter %.o,$(subst  .cc,.o,$(robot-sources)))
robot-objects       += $(filter %.o,$(subst .cpp,.o,$(robot-sources)))
robot-objects       += $(filter %.o,$(subst .cu,.o,$(robot-sources)))
robot-dependencies  := $(subst .o,.d,$(robot-objects))

mex-objects       := $(filter %.o,$(subst   .c,.o,$(mex-sources)))
mex-objects       += $(filter %.o,$(subst  .cc,.o,$(mex-sources)))
mex-objects       += $(filter %.o,$(subst .cpp,.o,$(mex-sources)))
mex-objects       += $(filter %.o,$(subst .cu,.o,$(mex-sources)))
mex-dependencies  := $(subst .o,.d,$(mex-objects))

sfunction-objects       := $(filter %.o,$(subst   .c,.o,$(sfunction-sources)))
sfunction-objects       += $(filter %.o,$(subst  .cc,.o,$(sfunction-sources)))
sfunction-objects       += $(filter %.o,$(subst .cpp,.o,$(sfunction-sources)))
sfunction-objects       += $(filter %.o,$(subst .cu,.o,$(sfunction-sources)))
sfunction-dependencies  := $(subst .o,.d,$(sfunction-objects))


libtarget     := $(libdir)/lib$(packagename).a
exetarget     := $(bindir)/$(packagename)
rtexetarget   := $(bindir)/RealTimeEDLUTKernel
steptarget     := $(bindir)/stepbystep
precisiontarget := $(bindir)/precisiontest
robottarget	:= $(bindir)/robottest
pkgconfigfile := $(packagename).pc

mextarget	:= $(mexdir)/$(packagename).$(mexsuffix)

sfunctiontarget	 := $(sfunctiondir)/$(packagename).$(mexsuffix)

automakefile := make.auto
