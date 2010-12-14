################################################################################
########################### - MAKEFILE FLAGS - #################################
################################################################################

CXXFLAGS += -I$(includedir) -DUSE_OPENCV

ifneq ($(external_sources),none)
 CXXFLAGS += `pkg-config --cflags ${external_sources}`
endif

ifneq ($(external_libraries),none)
 CXXFLAGS += `pkg-config --cflags ${external_libraries}`
 LDFLAGS  += `pkg-config --libs ${external_libraries}`
endif

ifeq ($(fortran77support),true)
 LDFLAGS += -lg2c
endif

ifeq ($(optimize),true)
  CXXFLAGS += -Wall -O3 -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF
  ifeq ($(specialize),true)
     CXXFLAGS += -march=$(platform) -mfpmath=sse
  endif
else
  CXXFLAGS += -g -Wall
endif

ifeq ($(parallelize),true)
    CXXFLAGS += -fopenmp
    CPPFLAGS += -fopenmp
endif

ifeq ($(sse-iset),true)
    CXXFLAGS += -msse -msse2
    CPPFLAGS += -msse -msse2
endif

ifeq ($(multi-threading),true)
    CXXFLAGS += -lpthread
endif

ifeq ($(profile),true)
  CXXFLAGS+= -pg
endif

ifeq ($(matlabsupport),true)
  CXXFLAGS	+= -I$(matlabinclude) -fPIC -ansi -pthread -DMATLAB_MEX_FILE
  MEXFLAGS	+= -cxx CC='$(compiler)' CXX='$(compiler)' LD='$(compiler)'
  LDFLAGS	+= 
endif

ifeq ($(simulinksupport),true)
  CXXFLAGS	+= -I$(simulinkinclude) -fPIC -ansi -pthread -DMATLAB_MEX_FILE
  MEXFLAGS	+= -cxx CC='$(compiler)' CXX='$(compiler)' LD='$(compiler)'
  LDFLAGS	+= 
endif

CXXFLAGS += -fno-strict-aliasing

ARFLAGS = ruv
CTAGFLAGS := -e -R --languages=c++,c

