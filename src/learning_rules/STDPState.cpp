/***************************************************************************
 *                           STDPState.cpp                                 *
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

#include "../../include/learning_rules/STDPState.h"

#include <cmath>

STDPState::STDPState(double NewLTPValue, double NewLTDValue): ConnectionState(2), LTPTau(NewLTPValue), LTDTau(NewLTDValue){
	ConnectionState::SetStateVariableAt(0,0); // Initialize presynaptic activity
	ConnectionState::SetStateVariableAt(1,0); // Initialize postsynaptic activity
}

STDPState::~STDPState() {
}

unsigned int STDPState::GetNumberOfPrintableValues(){
	return ConnectionState::GetNumberOfPrintableValues()+2;
}

double STDPState::GetPrintableValuesAt(unsigned int position){
	if (position<ConnectionState::GetNumberOfPrintableValues()){
		return ConnectionState::GetStateVariableAt(position);
	} else if (position==ConnectionState::GetNumberOfPrintableValues()) {
		return this->LTPTau;
	} else if (position==ConnectionState::GetNumberOfPrintableValues()+1) {
		return this->LTDTau;
	} else return -1;
}

//float STDPState::GetPresynapticActivity(){
//	return this->GetStateVariableAt(0);
//}

//float STDPState::GetPostsynapticActivity(){
//	return this->GetStateVariableAt(1);
//}

void STDPState::SetNewUpdateTime(double NewTime){
	float ElapsedTime=float(NewTime -  this->GetLastUpdateTime());

	float PreActivity = this->GetPresynapticActivity();
	float PostActivity = this->GetPostsynapticActivity();

	// Accumulate activity since the last update time
	PreActivity = PreActivity * exp(-ElapsedTime/this->LTPTau);
	PostActivity = PostActivity * exp(-ElapsedTime/this->LTDTau);

	// Store the activity in state variables
	this->SetStateVariableAt(0,PreActivity);
	this->SetStateVariableAt(1,PostActivity);

	this->SetLastUpdateTime(NewTime);
}

void STDPState::ApplyPresynapticSpike(){
	//float PreActivity = this->GetPresynapticActivity();

	//// Accumulate new incoming activity
	//PreActivity += 1;

	//// Store the activity in the state variable
	//this->SetStateVariableAt(0,PreActivity);


	// Increment the activity in the state variable
	this->SetStateVariableAt(0,this->GetPresynapticActivity()+1.0f);
}

void STDPState::ApplyPostsynapticSpike(){
	//float PostActivity = this->GetPostsynapticActivity();

	//// Accumulate new incoming activity
	//PostActivity += 1;

	//// Store the activity in the state variable
	//this->SetStateVariableAt(1,PostActivity);


	// Increment the activity in the state variable
	this->SetStateVariableAt(1,this->GetPostsynapticActivity()+1.0f);
}

