 #
 #                           CMakeLists.txt
 #                           --------------
 # copyright            : (C) 2018 by Jesus Garrido
 # email                : jesusgarrido@ugr.es

 #
 #
 #   This program is free software; you can redistribute it and/or modify
 #   it under the terms of the GNU General Public License as published by
 #   the Free Software Foundation; either version 3 of the License, or
 #   (at your option) any later version.
 #

 cmake_minimum_required( VERSION 3.5.1 )

 # add cmake modules: for all `include(...)` first look here
 list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

 project( edlut CXX C )
 set( EDLUT_USER_EMAIL "jesusgarrido@ugr.es" )

 # VERSION exported to config.h files, if exist
 set( EDLUT_VERSION_MAJOR 4 )
 set( EDLUT_VERSION_MINOR 0 )
 set( EDLUT_VERSION_PATCHLEVEL 0 )

 set( EDLUT_VERSION_VERSION "${EDLUT_VERSION_MAJOR}.${EDLUT_VERSION_MINOR}.${EDLUT_VERSION_PATCHLEVEL}" )
 set( EDLUT_VERSION_PRGNAME "${EDLUT_NAME}-${EDLUT_VERSION_VERSION}" )

 set(CMAKE_VERBOSE_MAKEFILE ON)

 ################################################################################
 ##################         All User Defined options           ##################
 ################################################################################

 # Set default settings and description
 set( with-cuda OFF CACHE STRING "Request compilation with CUDA parallelization. Optionally give directory with CUDA installation. [default=OFF]" )
 # Set parallelization scheme
 set( with-openmp ON CACHE STRING "Enable OpenMP multithreading. Optional: set OMP flag. [default=ON]" )
 # Set python bindings
 set( with-python ON CACHE STRING "Enable Python binding. Optional: set Python version (2 or 3). [default=ON]" )
 # additional compile flags
 set( with-optimize ON CACHE STRING "Enable user defined optimizations. [default ON, when ON, defaults to '-O3']" )
 set( with-debug OFF CACHE STRING "Enable user defined debug flags. [default OFF, when ON, defaults to '-g']" )
 set( with-profiling OFF CACHE STRING "Enable user defined profiling flags. [default OFF, when ON]" )
 # Generate Doxygen class documentation
 set( with-doc OFF CACHE STRING "Enable class documentation generation. [default OFF, when ON]" )
 # Compile in perfissive mode
 set( with-permissive ON CACHE STRING "Compile in permissive mode. [default ON]" )


 ################################################################################
 ##################   CPack, checks, ... in external modules   ##################
 ################################################################################

 # This include checks the symbols, etc.
 include( CheckIncludesSymbols )

 # These includes publish function names.
 include( ProcessUserOptions )

 # Must use GNUInstallDirs to install libraries into correct
 # locations on all platforms.
 include(GNUInstallDirs)

 ################################################################################
 ##################        Process user options                ##################
 ################################################################################

 process_with_optimize()
 process_with_debug()
 process_with_profiling()
 process_with_python()
 process_with_openmp()
 process_with_cuda()
 process_with_doc()
 process_with_permissive()


 ################################################################################
 ##################        Define source files                 ##################
 ################################################################################

 set ( edlut_library_sources
         ${all_sources}
         ${all_includes})

 set( edlut_kernel_source ./src/EDLUTKernel.cpp )
 set( edlut_stepbystep_source ./src/StepByStep.cpp )
 set( edlut_realtime_source ./src/RealTimeEDLUTKernel.cpp )

 set( edlut_cpp_api_source src/testCppAPI.cpp)


 ################################################################################
 ##################        Define compilation targets          ##################
 ################################################################################

 set( CMAKE_POSITION_INDEPENDENT_CODE ON )

 # EDLUT documentation
 if (DOC_FOUND)
   find_package(Doxygen)
 else()
   set (DOXYGEN_FOUND OFF)
 endif()

 # BOOST package
 FIND_PACKAGE( Boost 1.40 REQUIRED COMPONENTS)
 if (Boost_FOUND)
    INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
 endif()


 if (DOXYGEN_FOUND)
   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
   add_custom_target(doc
     ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
     COMMENT "Generating API documentation with Doxygen" VERBATIM
   )
 endif()

 # EDLUT library
 if (CUDA_FOUND)
   if( ("${CMAKE_VERSION}" VERSION_EQUAL 3.8) OR
        ("${CMAKE_VERSION}" VERSION_GREATER 3.8) )
        # Modern CMake
        set (MODERN_CMAKE ON)
        #enable_language("CUDA")
	enable_language(CUDA)
   else()
        set (MODERN_CMAKE OFF)
        find_package(CUDA 9.0 REQUIRED)
        if(NOT CUDA_FOUND)
            message(STATUS "CUDA not found")
        endif()
    endif()
	add_definitions(-DUSE_CUDA)
  endif()

  if (CUDA_FOUND)
    if ( NOT MODERN_CMAKE)
      cuda_add_library( ${PROJECT_NAME} "")
    else()
      add_library( ${PROJECT_NAME} "")
    endif()

    set_target_properties(${PROJECT_NAME}
                        PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties( ${PROJECT_NAME}
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${PROJECT_NAME}
			PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    ############################################# SET CUDA ARCH ######################################################
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_20,code=sm_20>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_21,code=sm_21>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_30,code=sm_30>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_32,code=sm_32>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_35,code=sm_35>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_52,code=sm_52>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_53,code=sm_53>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_60>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_62,code=sm_62>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_72,code=sm_72>)
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)

    #Visual Studio (MSVC) is able to automatically detect the CUDA GPU ARCHITECTURE. In other case, we must select this value
    #executing the auxiliar code "cuda_arch".
    if(NOT DEFINED MSVC)
      #Auxiliar code to automaticaly check the cuda architecture of all the GPUs
      set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda_arch.cu)
      set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda_arch) 
      #Compile the Auxiliar code
      execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -ccbin=${CMAKE_CXX_COMPILER} ${CUDAFILE} -o ${OUTPUTFILE})

      #Execute the Auxiliar code to obtain the GPU architecture.
      execute_process(COMMAND ${OUTPUTFILE} RESULT_VARIABLE CUDA_RETURN_CODE OUTPUT_VARIABLE CUDA_ARCH_OUTPUT)

      message("CUDA_ARCH_OUTPUT = ${CUDA_ARCH_OUTPUT}")
      #If there is CUDA capable GPUs and we are not using visual studio
      if(${CUDA_RETURN_CODE} EQUAL 1)
        target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode ${CUDA_ARCH_OUTPUT}>)
      endif()
    endif()

  else()
    add_library( ${PROJECT_NAME} "")
  endif()

#target_include_directories (lib${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/include")
 target_include_directories( ${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

 # Kernel standalone application
 add_executable( ${PROJECT_NAME}kernel ${edlut_kernel_source})
 target_link_libraries (${PROJECT_NAME}kernel ${PROJECT_NAME})

 # StepByStep test application
 add_executable( ${PROJECT_NAME}_step ${edlut_stepbystep_source})
 target_link_libraries (${PROJECT_NAME}_step ${PROJECT_NAME})

 # Real-time test application
 add_executable( ${PROJECT_NAME}_realtime ${edlut_realtime_source})
 target_link_libraries (${PROJECT_NAME}_realtime ${PROJECT_NAME})

 # StepByStep test application
 add_executable( ${PROJECT_NAME}_cpp_api_example ${edlut_cpp_api_source})
 target_link_libraries (${PROJECT_NAME}_cpp_api_example ${PROJECT_NAME})

 ################################################################################
 ##################        Define Subdirectories here          ##################
 ################################################################################
 add_subdirectory(src)
 add_subdirectory(include)
 if ( HAVE_PYTHON )
     add_subdirectory( python )
 endif ()



 # ################################################################################
 ##################        Installation instructions           ##################
 ################################################################################

 install ( TARGETS ${PROJECT_NAME}kernel DESTINATION ${CMAKE_INSTALL_BINDIR})
 install ( TARGETS ${PROJECT_NAME}_step DESTINATION ${CMAKE_INSTALL_BINDIR})
 install ( TARGETS ${PROJECT_NAME}_realtime DESTINATION ${CMAKE_INSTALL_BINDIR})
 install ( TARGETS ${PROJECT_NAME}_cpp_api_example DESTINATION ${CMAKE_INSTALL_BINDIR})

 install(TARGETS ${PROJECT_NAME}
     ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
     LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
     RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
 install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


 ##INSTALL THE SCRIPT REQUIRED TO EXPORT THE ENVIRONMENT VARIABLES ON WINDOWS
 if(DEFINED WIN32)
  install( PROGRAMS
         ${PROJECT_BINARY_DIR}/vars.ps1
         DESTINATION ${CMAKE_INSTALL_BINDIR}
         )
 ##INSTALL THE SCRIPT REQUIRED TO EXPORT THE ENVIRONMENT VARIABLES ON LINUX
 else()
  install( PROGRAMS
         ${PROJECT_BINARY_DIR}/vars.sh
         DESTINATION ${CMAKE_INSTALL_BINDIR}
         )
 endif()


 if (DOXYGEN_FOUND)
   install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html DESTINATION ${CMAKE_INSTALL_DOCDIR})
 endif()


 ################################################################################
 ##################           File generation here             ##################
 ################################################################################

 configure_file(
         "${PROJECT_SOURCE_DIR}/python/setup.py.in"
         "${PROJECT_BINARY_DIR}/python/setup.py" @ONLY
 )

 ##GENERATE THE SCRIPT REQUIRED TO EXPORT THE ENVIRONMENT VARIABLES ON WINDOWS
 if(DEFINED WIN32)
   configure_file(
         "${PROJECT_SOURCE_DIR}/utils/vars.ps1.in"
         "${PROJECT_BINARY_DIR}/vars.ps1" @ONLY
   )
 ##GENERATE THE SCRIPT REQUIRED TO EXPORT THE ENVIRONMENT VARIABLES ON LINUX
 else()
   configure_file(
         "${PROJECT_SOURCE_DIR}/utils/vars.sh.in"
         "${PROJECT_BINARY_DIR}/vars.sh" @ONLY
   )
 endif()

