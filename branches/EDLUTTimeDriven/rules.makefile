################################################################################
################################ - MAKEFILE RULES - ############################
################################################################################

compiler	:= g++
mex		:= mex
CXX 		:= ${compiler}

.PHONY         : $(exetarget)
$(exetarget) : $(exe-objects)
	@echo compiler path = ${compiler}
	@echo
	@echo ------------------ making executable
	@echo
	@mkdir -p $(bindir)
	$(compiler) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

.PHONY		: mex
mex	: $(mextarget) 
	@echo
	@echo ------------------ making mex file $(mextarget)
	@echo

$(mextarget)	: $(mex-objects)
	@echo compiler path = ${mex}
	@echo
	@echo ------------------ making mexfile
	@echo
	@mkdir -p $(mexdir)
	$(mex) $(MEXFLAGS) $^ $(LDFLAGS) -o $@
	
.PHONY		: sfunction
sfunction	: $(sfunctiontarget) 
	@echo
	@echo ------------------ making sfunction file $(sfunctiontarget)
	@echo

$(sfunctiontarget)	: $(sfunction-objects)
	@echo compiler path = ${mex}
	@echo
	@echo ------------------ making sfunction file
	@echo
	@mkdir -p $(sfunctiondir)
	$(mex) $(MEXFLAGS) $^ $(LDFLAGS) -o $@

.PHONY  : library
library : $(libtarget)
	@echo
	@echo ------------------ making library $(libtarget)
	@echo

$(libtarget): $(objects)
	@echo
	@echo ------------------ creating library
	@echo
	@mkdir -p $(libdir)
	$(AR) $(ARFLAGS) $@ $^

.PHONY : tags
tags   :
	@echo
	@echo ------------------ creating tag entries
	@echo
	@etags $(includedir)/*.h $(includedir)/*.h $(srcdir)/*.cpp $(srcdir)/*.cc $(srcdir)/*.c

.PHONY : dox
dox    : Doxyfile
	@echo
	@echo ------------------ creating documentation
	@echo
	@doxygen Doxyfile

.PHONY   : doxclean
doxclean :
	@echo
	@echo ------------------ removing documentation
	@echo
	@rm -rf doc

.PHONY : distclean
distclean  :
	@echo
	@echo ------------------ cleaning everything
	@echo
	@rm -f $(pkgconfigfile) $(libtarget) $(packagename) $(objects) ${exetarget}.exe ${exe-objects} $(dependencies) ${exe-dependencies} ${mextarget} ${mex-objects} {mex-dependencies} ${sfunctiontarget} ${sfunction-objects} {sfunction-dependencies} TAGS gmon.out

.PHONY : clean
clean  :
	@echo
	@echo ------------------ cleaning *.o exe lib
	@echo
	@rm -f $(objects) ${exe-objects} ${libtarget} ${exetarget}.exe ${mextarget} ${mex-objects} ${sfunctiontarget} ${sfunction-objects} TAGS gmon.out

.PHONY : clear
clear :
	@rm -rf \#* ${dependencies}

.PHONY: install
install: $(libtarget) pkgfile uninstall
	@echo
	@echo ------------------ installing library and header files
	@echo
	@echo ------------------ installing at $(installdir)
	@echo
	@mkdir -p $(installdir)/include/$(packagename)
	@cp -vfR $(includedir)/[!.]* $(installdir)/include/$(packagename)
	@mkdir -p $(installdir)/lib/pkgconfig
	@cp -vfR $(libtarget)  $(installdir)/lib
	@echo
	@echo ------------------ installing the pkg-config file to $(installdir)/lib/pkgconfig. \
		Remember to add this path to your PKG_CONFIG_PATH variable
	@echo
	@cp $(pkgconfigfile) $(installdir)/lib/pkgconfig/

.PHONY: install-exe
install-exe: $(exetarget)
	@cp $(exetarget) $(installdir)/bin

.PHONY: install-dev
install-dev : $(libtarget) pkgfile uninstall
	@echo
	@echo ------------------ installing library and development files
	@echo
	@echo ------------------ installing at $(installdir)
	@echo
	@mkdir -p $(installdir)/include/$(packagename)
	@cp -vfR $(includedir)/$(packagename)/[!.]* $(installdir)/include/$(packagename)
	@mkdir -p $(installdir)/lib/pkgconfig
	@cp -vfR $(libtarget)  $(installdir)/lib                 # copy the static library
	@mkdir -p $(installdir)/src/$(packagename)                 # create the source directory
	@cp -vfR $(srcdir)/*.c* $(installdir)/src/$(packagename) # copy development files
	@cp -vf makefile $(installdir)/src/$(packagename)
	@cp $(pkgconfigfile) $(installdir)/lib/pkgconfig/

.PHONY: uninstall
uninstall:
	@echo
	@echo ------------------ uninstalling if-installed
	@echo
	@rm -rf $(installdir)/include/$(packagename)
	@rm -f   $(installdir)/$(libtarget)
	@rm -rf $(installdir)/src/$(packagename)
	@rm -f   $(installdir)/lib/pkgconfig/$(pkgconfigfile)
	@rm -f   $(installdir)/bin/$(exetarget)

ifneq "$(MAKECMDGOALS)" "clean"
  include $(dependencies)
endif

%.d : %.cc
	@echo
	@echo ------------------ creating dependencies for $@
	@echo
	$(compiler) $(CXXFLAGS) $(TARGET_ARCH) -MM $< | \
	sed 's,\($(notdir $*)\.o\) *:,$(dir $@)\1 $@: ,' > $@.tmp
	mv -f $@.tmp $@
	@echo

%.d : %.cpp
	@echo
	@echo ------------------ creating dependencies for $@
	@echo
	$(compiler) $(CXXFLAGS) $(TARGET_ARCH) -MM $< | \
	sed 's,\($(notdir $*)\.o\) *:,$(dir $@)\1 $@: ,' > $@.tmp
	mv -f $@.tmp $@
	@echo

.PHONY : pkgfile
pkgfile:
	@echo
	@echo ------------------ creating pkg-config file
	@echo
	@echo "# Package Information for pkg-config"    >  $(pkgconfigfile)
	@echo "# Author: $(author)" 			>> $(pkgconfigfile)
	@echo "# Created: `date`"			>> $(pkgconfigfile)
	@echo "# Licence: $(licence)"			>> $(pkgconfigfile)
	@echo 						>> $(pkgconfigfile)
	@echo prefix=$(installdir)       		>> $(pkgconfigfile)
	@echo exec_prefix=$$\{prefix\}     		>> $(pkgconfigfile)
	@echo libdir=$$\{exec_prefix\}/lib 		>> $(pkgconfigfile)
	@echo includedir=$$\{prefix\}/include   	>> $(pkgconfigfile)
	@echo 						>> $(pkgconfigfile)
	@echo Name: "$(packagename)" 			>> $(pkgconfigfile)
	@echo Description: "$(description)" 		>> $(pkgconfigfile)
	@echo Version: "$(version)" 			>> $(pkgconfigfile)
	@echo Libs: -L$$\{libdir} -l$(packagename) 	>> $(pkgconfigfile)
	@echo Cflags: -I$$\{includedir\} 		>> $(pkgconfigfile)
	@echo 						>> $(pkgconfigfile)

.PHONY : revert
revert :
	@mv -f makefile.in makefile


.PHONY : export
export :
	@echo "#automatically generated makefile"         >  $(automakefile)
	@echo packagename := ${packagename}               >> ${automakefile}
	@echo version := ${version}                       >> ${automakefile}
	@echo author := ${author}                         >> ${automakefile}
	@echo description := "${description}"             >> ${automakefile}
	@echo licence := ${licence}                       >> ${automakefile}
	@echo "#........................................" >> ${automakefile}
	@echo installdir := ${installdir}                 >> $(automakefile)
	@echo external_sources := ${external_sources}     >> ${automakefile}
	@echo external_libraries := ${external_libraries} >> ${automakefile}
	@echo libdir := ${libdir}                         >> ${automakefile}
	@echo srcdir := ${srcdir}                         >> ${automakefile}
	@echo includedir:= ${includedir}                  >> ${automakefile}
	@echo "#........................................" >> ${automakefile}
	@echo optimize := ${optimize}                     >> ${automakefile}
	@echo fortran77support := ${fortran77support}     >> ${automakefile}
	@echo sse-iset := ${sse-iset}                     >> ${automakefile}
	@echo multi-threading := ${multi-threading}       >> ${automakefile}
	@echo parallelize := ${parallelize}               >> ${automakefile}
	@echo profile := ${profile}                       >> ${automakefile}
	@echo "#........................................" >> ${automakefile}
	@echo specialize := ${specialize}                 >> ${automakefile}
	@echo platform := ${platform}                     >> ${automakefile}
	@echo "#........................................" >> ${automakefile}
	@echo exe-source-file := ${exe-source-file}       >> ${automakefile}
	@echo sources := ${sources}                       >> ${automakefile}
	@echo                                             >> ${automakefile}
	@cat ${MAKEFILE_HEAVEN}/static-variables.makefile >> ${automakefile}
	@cat ${MAKEFILE_HEAVEN}/flags.makefile            >> ${automakefile}
	@cat ${MAKEFILE_HEAVEN}/rules.makefile            >> ${automakefile}
	@echo >> ${automakefile}
	@mv makefile makefile.in
	@mv ${automakefile} makefile

.PHONY : flags
flags :
	@echo
	@echo ------------------ build flags
	@echo
	@echo ldflags  = $(LDFLAGS)
	@echo cxxflags = $(CXXFLAGS)
	@echo mexflags = $(MEXFLAGS)
	@echo sources = ${sources}
	@echo objects = ${exe-objects}


.PHONY : gflat
gflat :
	@gprof $(packagename) gmon.out -p | more

.PHONY : gcall
gcall :
	@gprof $(packagename) gmon.out -q | more

.PHONY : state
state  :
	@echo
	@echo "package name      : ${packagename} v${version} by ${author}"
	@echo "                   (${description}) "
	@echo "------------------------------------------------------------------------"
	@echo "install directory : ${installdir}"
	@echo "external sources  : ${external_sources}"
	@echo "external libs     : ${external_libraries}"
	@echo "fortran support   : ${fortran77support}"
	@echo "------------------------------------------------------------------------"
	@echo "optimize          : ${optimize}"
	@echo "parallelize       : ${parallelize}"
	@echo "profile           : ${profile}"
	@echo "sse-iset          : ${sse-iset}"
	@echo "multi-threading   : ${multi-threading}"
	@echo "------------------------------------------------------------------------"
	@echo "specialize        : ${specialize} for ${platform}"
	@echo "------------------------------------------------------------------------"
	@echo "sources           : ${sources}"
	@echo "------------------------------------------------------------------------"
	@echo ldflags  = $(LDFLAGS)
	@echo cxxflags = $(CXXFLAGS)
	@echo sources = ${sources}
	@echo objects = ${exe-objects}

.PHONY : rules
rules :
	@echo
	@echo ------------------ legitimate rules
	@echo
	@echo "(nothing)   : makes the executable : by default src/main.cpp is included to the sources list"
	@echo "              and used in the exe-build. Change its value with $exe-source-file variable"
	@echo "library     : generates the library"
	@echo "tags        : generates etags files"
	@echo "dox         : generates the doxygen documentation if Doxyfile exists"
	@echo "doxclean    : cleans up the documentation"
	@echo "clean       : cleans up .o lib and exe files"
	@echo "distclean   : cleans everything except source+headers"
	@echo "install     : installs the library"
	@echo "install-dev : installs the library along with documentation files"
	@echo "install-exe : installs the executable"
	@echo "uninstall   : uninstalls the library"
	@echo "pkgfile     : generates the pkg-config file"
	@echo "flags       : shows the flags that will be used"
	@echo "gflat       : shows gprof profiler flat view result"
	@echo "gcall       : shows gprof profiler call graph view result"
	@echo "rules       : shows this text"
	@echo "state       : show the configuration state of the package"
	@echo "export      : export the makefile"
	@echo "revert      : moves makefile.in to makefile"
	@echo "clear       : clears #* & dependency files"
	@echo


