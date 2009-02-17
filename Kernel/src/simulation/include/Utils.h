#ifndef UTILS_H_
#define UTILS_H_

/*!
 * \file Utils.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares functions for read and processs files.
 */

#include <cstdio>

#define COMMENT_CHAR '/'

/*!
 * \brief It skips the white lines, spaces and tab from the Currentline.
 * 
 * It skips the white lines from the Currentline in advance.
 * 
 * \param fh The file handler.
 * \param Currentline The currentline where we are readding.
 * 
 * \return True if white lines have been correctly cleared. False in other case.
 */
int skip_spaces(FILE *fh, long & Currentline);

/*!
 * \brief It skips the comment lines from the Currentline.
 * 
 * It skips the comment lines from the Currentline in advance.
 * 
 * \param fh The file handler.
 * \param Currentline The currentline where we are readding.
 */  
void skip_comments(FILE *fh, long & Currentline);
  
#endif /*UTILS_H_*/
