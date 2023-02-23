 #
 #                           CMakeLists.txt
 #                           --------------
 # copyright            : (C) 2019 by Jesus Garrido
 # email                : jesusgarrido@ugr.es
 #
 #
 #   This program is free software; you can redistribute it and/or modify
 #   it under the terms of the GNU General Public License as published by
 #   the Free Software Foundation; either version 3 of the License, or
 #   (at your option) any later version.
 #

if ( HAVE_PYTHON )

  if ( CYTHON_FOUND )
    include( UseCython )
    set_source_files_properties(
        simulation_wrapper.pyx
        PROPERTIES CYTHON_IS_CXX TRUE )

    cython_add_module( simulation_wrapper simulation_wrapper.pyx )
  else ()
    message( FATAL_ERROR "You have configured EDLUT to build the Python bindings PyEDLUT, but "
        "there is no Cython version found on your system. Install Cython (>=0.19.2):\n")
  endif ()

  target_link_libraries( simulation_wrapper ${PROJECT_NAME})

  target_include_directories( simulation_wrapper PRIVATE
      ${PROJECT_SOURCE_DIR}/include
      ${PYTHON_INCLUDE_DIRS}
          )

  target_compile_definitions( simulation_wrapper PRIVATE
      -D_IS_PYEDLUT
      )

  ########### IT IS REQUIRED TO CHANGE THE PATH FOR WINDOWS. ###########
  if(DEFINED WIN32)
    STRING(REGEX REPLACE ":/" ":\\\\" BUILD_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/build)
    STRING(REGEX REPLACE ":/" ":\\\\" PREFIX_DIR ${CMAKE_INSTALL_PREFIX})
    STRING(REGEX REPLACE ":/" ":\\\\" INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/${PYEXECDIR})
    STRING(REGEX REPLACE ":/" ":\\\\" INSTALL_SCRIPT_DIR ${CMAKE_INSTALL_FULL_BINDIR})
    STRING(REGEX REPLACE ":/" ":\\\\" INSTALL_DATA_DIR ${CMAKE_INSTALL_FULL_DATADIR})
    install( CODE "execute_process(
      COMMAND ${PYTHON} setup.py build --build-base=${BUILD_BASE_DIR}
                               install --prefix=${PREFIX_DIR}
                                       --install-lib=${INSTALL_LIB_DIR}
                                       --install-scripts=${INSTALL_SCRIPT_DIR}
                                       --install-data=${INSTALL_DATA_DIR}
      WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")"
    )
  else()
    install( CODE "execute_process(
      COMMAND ${PYTHON} setup.py build --build-base=${CMAKE_CURRENT_BINARY_DIR}/build
                               install --prefix=${CMAKE_INSTALL_PREFIX}
                                       --install-lib=${CMAKE_INSTALL_PREFIX}/${PYEXECDIR}
                                       --install-scripts=${CMAKE_INSTALL_FULL_BINDIR}
                                       --install-data=${CMAKE_INSTALL_FULL_DATADIR}
      WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")"
    )
  endif()


  install( TARGETS simulation_wrapper DESTINATION ${PYEXECDIR}/pyedlut/ )
endif ()
