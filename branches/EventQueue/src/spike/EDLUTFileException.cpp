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

#include "../../include/spike/EDLUTFileException.h"

EDLUTFileException::EDLUTFileException(int a, int b, int c, int d, long Line): EDLUTException(a,b,c,d), Currentline(Line) {
}

long EDLUTFileException::GetErrorLine() const {
	return this->Currentline;
}

void EDLUTFileException::display_error() const {

	char msgbuf[160];
	if(this->GetErrorNum()){
		cerr << "Error while: " << this->GetTaskMsg() << endl;
		if((this->GetErrorNum() & 0xFF) == 1){
			cerr << "In file line: " << Currentline << endl;
		}
		
		sprintf(msgbuf,"Error message (%08lX): %s",this->GetErrorNum(),this->GetErrorMsg());
		cerr << msgbuf << endl;
		cerr << "Try to: " << this->GetRepairMsg() << endl;
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

