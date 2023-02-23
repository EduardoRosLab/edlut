#
#                           CheckIncludesSymbols.cmake
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

include( CheckIncludeFiles )
check_include_files( "stdint.h" HAVE_STDINT_H )
check_include_files( "arpa/inet.h" HAVE_INET_H )
check_include_files( "fcntl.h" HAVE_FCNTL_H )
check_include_files( "limits.h" HAVE_LIMITS_H )
check_include_files( "netdb.h" HAVE_NETDB_H )
check_include_files( "string.h" HAVE_STRING_H )
check_include_files( "sys/socket.h" HAVE_SOCKET_H )
check_include_files( "unistd.h" HAVE_UNISTD_H )
check_include_files( "wchar.h" HAVE_WCHAR_H )

include( CheckIncludeFileCXX )
check_include_file_cxx( "istream" HAVE_ISTREAM )
check_include_file_cxx( "ostream" HAVE_OSTREAM )
check_include_file_cxx( "sstream" HAVE_SSTREAM )

# Check types exist
include( CheckTypeSize )
check_type_size( int16_t INT16_T_SIZE ) # also sets HAVE_LONG_LONG
if ( INT16_T_SIZE GREATER 0 )
  set( HAVE_INT16_T ON )
endif ()

check_type_size( int32_t INT32_T_SIZE ) # also sets HAVE_LONG_LONG
if ( INT32_T_SIZE GREATER 0 )
  set( HAVE_INT32_T ON )
endif ()

check_type_size( int64_t INT64_T_SIZE ) # also sets HAVE_LONG_LONG
if ( INT64_T_SIZE GREATER 0 )
  set( HAVE_INT64_T ON )
endif ()

check_type_size( int8_t INT8_T_SIZE ) # also sets HAVE_LONG_LONG
if ( INT8_T_SIZE GREATER 0 )
  set( HAVE_INT8_T ON )
endif ()

check_type_size( uint32_t UINT32_T_SIZE ) # also sets HAVE_LONG_LONG
if ( UINT32_T_SIZE GREATER 0 )
  set( HAVE_UINT32_T ON )
endif ()

check_type_size( uint16_t UINT16_T_SIZE ) # also sets HAVE_LONG_LONG
if ( UINT16_T_SIZE GREATER 0 )
  set( HAVE_UINT16_T ON )
endif ()

check_type_size( uint64_t UINT64_T_SIZE ) # also sets HAVE_LONG_LONG
if ( UINT64_T_SIZE GREATER 0 )
  set( HAVE_UINT64_T ON )
endif ()

check_type_size( uint8_t UINT8_T_SIZE ) # also sets HAVE_LONG_LONG
if ( UINT8_T_SIZE GREATER 0 )
  set( HAVE_UINT8_T ON )
endif ()

# Check functions exist
include( CheckFunctionExists )
check_function_exists( malloc "stdlib.h" HAVE_MALLOC )
check_function_exists( memset "stdlib.h" HAVE_MEMSET )
check_function_exists( pow "math.h" HAVE_POW )
check_function_exists( socket "sys/socket.h" HAVE_SOCKET )
check_function_exists( sqrt "math.h" HAVE_SQRT )
