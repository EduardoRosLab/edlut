/***************************************************************************
 *                           Utils.cpp                                     *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 
#include "../../include/simulation/Utils.h"

int skip_spaces(FILE *fh, long & Currentline){
	int ch;
	while((ch=fgetc(fh)) == ' ' || ch=='\r' || ch=='\n') // take all spaces
		if(ch=='\n')
			Currentline++;
			
	return(ch);
}

void skip_comments(FILE *fh, long & Currentline){
	int ch;
	while((ch=skip_spaces(fh, Currentline)) == COMMENT_CHAR){
		while((ch=fgetc(fh)) != EOF && ch != '\n');
		
		if(ch=='\n')
			Currentline++;
    }
   
   	if(ch != EOF)
   		ungetc(ch, fh);
}
