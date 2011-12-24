/***************************************************************************
 *                           ConnectionState.cpp                           *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
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

#include "../../include/learning_rules/ConnectionState.h"

ConnectionState::ConnectionState(unsigned int NumVariables): NumberOfVariables(NumVariables), LastUpdate(0){
	// TODO Auto-generated constructor stub
	this->StateVars = (float *) new float [NumVariables];
}

ConnectionState::~ConnectionState() {
	// TODO Auto-generated destructor stub
	if (this->StateVars!=0){
		delete [] this->StateVars;
	}
}

void ConnectionState::SetStateVariableAt(unsigned int position,float NewValue){
	this->StateVars[position] = NewValue;
}

unsigned int ConnectionState::GetNumberOfVariables(){
	return this->NumberOfVariables;
}

float ConnectionState::GetStateVariableAt(unsigned int position){
	return this->StateVars[position];
}

double ConnectionState::GetLastUpdateTime(){
	return this->LastUpdate;
}

void ConnectionState::SetLastUpdateTime(double NewUpdateTime){
	this->LastUpdate = NewUpdateTime;
}

unsigned int ConnectionState::GetNumberOfPrintableValues(){
	return this->GetNumberOfVariables()+1;
}

double ConnectionState::GetPrintableValuesAt(unsigned int position){
	if (position<this->GetNumberOfVariables()){
		return this->GetStateVariableAt(position);
	} else if (position==this->GetNumberOfVariables()) {
		return this->GetLastUpdateTime();
	} else return -1;
}




