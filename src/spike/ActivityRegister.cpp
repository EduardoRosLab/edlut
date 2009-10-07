/***************************************************************************
 *                           ActivityRegister.cpp                          *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido                        *
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

#include "../../include/spike/ActivityRegister.h"


ActivityRegister::ActivityRegister(int VarNumber): numvar(VarNumber) {
	if (VarNumber>0){
		values = new float[VarNumber];			
	}else{
		values = 0;
	}
}

ActivityRegister::~ActivityRegister(){
	if (values!=0){
		delete [] values;
	}	
}

int ActivityRegister::GetVarNumber() const{
	return numvar;
}

float ActivityRegister::GetVarValueAt(unsigned int index) const{
	return values[index];
}
		
void ActivityRegister::SetVarValueAt(unsigned int index, float value){
	values[index]=value;
}
		
		
		
