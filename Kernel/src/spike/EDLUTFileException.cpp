/***************************************************************************
 *                           EDLUTFileException.cpp                        *
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

#include "./include/EDLUTFileException.h"

EDLUTFileException::EDLUTFileException(int a, int b, int c, int d, long Line): EDLUTException(a,b,c,d), Currentline(Line) {
}

long EDLUTFileException::GetErrorLine() const {
	return this->Currentline;
}

void EDLUTFileException::display_error() const {

	char msgbuf[160];
	if(this->GetErrorNum()){
		sprintf(msgbuf,"Error while: %s",this->GetTaskMsg());
		fprintf(stderr,msgbuf);
		if((this->GetErrorNum() & 0xFF) == 1){
			sprintf(msgbuf,"In file line: %li",Currentline);
			fprintf(stderr,msgbuf);
		}
		
		sprintf(msgbuf,"Error message (%08lX): %s",this->GetErrorNum(),this->GetErrorMsg());
		fprintf(stderr,msgbuf);
		sprintf(msgbuf,"Try to: %s",this->GetRepairMsg());
		fprintf(stderr,msgbuf);
	}
}

ostream & operator<< (ostream & out, EDLUTFileException Exception){
	if(Exception.GetErrorNum()){
		out << "Error while: " << Exception.GetTaskMsg() << endl;
		if((Exception.GetErrorNum() & 0xFF) == 1){
			out << "In file line: " << Exception.GetErrorLine() << endl;
		}
		
		out << "Error message " << Exception.GetErrorNum() << ": " << Exception.GetErrorMsg() << endl;
		out << "Try to: " << Exception.GetRepairMsg() << endl;
	}
	
	return out;
}

