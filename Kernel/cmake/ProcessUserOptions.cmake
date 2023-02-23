#
#                           ProcessUserOptions.cmake
#                           --------------------------
# copyright            : (C) 2018 by Jesus Garrido
# email                : jesusgarrido@ugr.es

#
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 3 of the License, or
#   (at your option) any later version.
#

function( PROCESS_WITH_OPTIMIZE )
  if ( with-optimize )
    if ( with-optimize STREQUAL "ON" )
      if(WIN32)
        set( with-optimize "-O2 -DHAVE_INLINE" )
	set( with-optimize_CPU "-GL -Ot" )
      else()
        set( with-optimize "-O3 -DHAVE_INLINE" )
      endif ()
    endif ()
    foreach ( flag ${with-optimize} )
      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE )
      set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${flag}" PARENT_SCOPE)
    endforeach ()
    foreach ( flag ${with-optimize_CPU} )
      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE )
    endforeach ()
  endif ()
endfunction()

function( PROCESS_WITH_DEBUG )
  if ( with-debug )
    if ( with-debug STREQUAL "ON" )
      set( with-debug "-g" )
    endif ()
    foreach ( flag ${with-debug} )
      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE )
      set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${flag}" PARENT_SCOPE)
    endforeach ()
  endif ()
endfunction()

function( PROCESS_WITH_PROFILING )
  if ( with-profiling )
    if ( with-profiling STREQUAL "ON" )
      set( with-profiling "-lprofiler" )
    endif ()
    foreach ( flag ${with-profiling} )
      set( CMAKE_LDFLAGS "${CMAKE_LDFLAGS} ${flag}" PARENT_SCOPE )
    endforeach ()
  endif ()
endfunction()

function( PROCESS_WITH_OPENMP )
  # Find OPENMP
  if ( with-openmp )
    if ("${with-openmp}" STREQUAL "OFF")
      # set variables in this scope
      set( OPENMP_FOUND OFF PARENT_SCOPE)
      set( OpenMP_C_FLAGS "" PARENT_SCOPE)
      set( OpenMP_CXX_FLAGS "" PARENT_SCOPE)
    elseif ( NOT "${with-openmp}" STREQUAL "ON" )
      message( STATUS "Set OpenMP argument: ${with-openmp}")
      # set variables in this scope
      set( OPENMP_FOUND ON PARENT_SCOPE)
      set( OpenMP_C_FLAGS "${with-openmp}" PARENT_SCOPE)
      set( OpenMP_CXX_FLAGS "${with-openmp}" PARENT_SCOPE)
    else ()
      find_package( OpenMP )
    endif ()
    if ( OPENMP_FOUND )
      # export found variables to parent scope
      set( OPENMP_FOUND "${OPENMP_FOUND}" PARENT_SCOPE )
      set( OpenMP_C_FLAGS "${OpenMP_C_FLAGS}" PARENT_SCOPE )
      set( OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS}" PARENT_SCOPE )
      # set flags
      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" PARENT_SCOPE )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" PARENT_SCOPE )
      set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}" PARENT_SCOPE)
    endif ()
  endif ()
endfunction()

function( PROCESS_WITH_CUDA )
  # Find OPENMP
  if ( with-cuda )
    if ("${with-cuda}" STREQUAL "OFF")
      # set variablesin this scope
      set( CUDA_FOUND OFF PARENT_SCOPE)
    elseif ( "${with-cuda}" STREQUAL "ON" )
      set( CUDA_FOUND ON PARENT_SCOPE)
    else ()
      message( STATUS "Invalid with-cuda argument ${with-cuda}. Only ON/OFF are allowed")
      set( CUDA_FOUND OFF PARENT_SCOPE)
    endif ()
  endif ()
endfunction()

function( PROCESS_WITH_DOC )
  # Find OPENMP
  if ( with-doc )
    if ("${with-doc}" STREQUAL "OFF")
      # set variablesin this scope
      set( DOC_FOUND OFF PARENT_SCOPE)
    elseif ( "${with-doc}" STREQUAL "ON" )
      set( DOC_FOUND ON PARENT_SCOPE)
    else ()
      message( STATUS "Invalid with-doc argument ${with-doc}. Only ON/OFF are allowed")
      set( DOC_FOUND OFF PARENT_SCOPE)
    endif ()
  endif ()
endfunction()

function( PROCESS_WITH_PYTHON )
  # Find Python
  set( HAVE_PYTHON OFF PARENT_SCOPE )
  if ( ${with-python} STREQUAL "ON" OR  ${with-python} STREQUAL "2" OR  ${with-python} STREQUAL "3" )

    # Localize the Python interpreter
    if ( ${with-python} STREQUAL "ON" )
      find_package( PythonInterp )
    elseif ( ${with-python} STREQUAL "2" )
      find_package( PythonInterp 2 REQUIRED )
    elseif ( ${with-python} STREQUAL "3" )
      find_package( PythonInterp 3 REQUIRED )
    endif ()

    if ( PYTHONINTERP_FOUND )
      set( PYTHONINTERP_FOUND "${PYTHONINTERP_FOUND}" PARENT_SCOPE )
      set( PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE )
      set( PYTHON ${PYTHON_EXECUTABLE} PARENT_SCOPE )
      set( PYTHON_VERSION ${PYTHON_VERSION_STRING} PARENT_SCOPE )

      # Localize Python lib/header files and make sure that their version matches
      # the Python interpreter version !
      find_package( PythonLibs ${PYTHON_VERSION_STRING} EXACT )
      if ( PYTHONLIBS_FOUND )
        set( HAVE_PYTHON ON PARENT_SCOPE )
        # export found variables to parent scope
        set( PYTHONLIBS_FOUND "${PYTHONLIBS_FOUND}" PARENT_SCOPE )
        set( PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIRS}" PARENT_SCOPE )
        set( PYTHON_LIBRARIES "${PYTHON_LIBRARIES}" PARENT_SCOPE )

        find_package( Cython )
        if ( CYTHON_FOUND )
          # confirmed not working: 0.15.1
          # confirmed working: 0.19.2+
          # in between unknown
          if ( CYTHON_VERSION VERSION_LESS "0.19.2" )
            message( FATAL_ERROR "Your Cython version is too old. Please install "
                    "newer version (0.19.2+)" )
          endif ()

          # export found variables to parent scope
          set( CYTHON_FOUND "${CYTHON_FOUND}" PARENT_SCOPE )
          set( CYTHON_EXECUTABLE "${CYTHON_EXECUTABLE}" PARENT_SCOPE )
          set( CYTHON_VERSION "${CYTHON_VERSION}" PARENT_SCOPE )
        endif ()
        set( PYEXECDIR "${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages" PARENT_SCOPE )
      endif ()
    endif ()
  elseif ( ${with-python} STREQUAL "OFF" )
  else ()
    message( FATAL_ERROR "Invalid option: -Dwith-python=" ${with-python} )
  endif ()
endfunction()


function( PROCESS_WITH_PERMISSIVE )
  if ( with-permissive )
    if ( with-permissive STREQUAL "ON" )
      set( with-permissive "-fpermissive" )
    endif ()
    foreach ( flag ${with-permissive} )
      set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE )
#      set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${flag}" PARENT_SCOPE)
    endforeach ()
  endif ()
endfunction()

