/***************************************************************************
 *                           STDPLSState.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Jesus Garrido                        *
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

#include "../../include/learning_rules/STDPLSState.h"

#include <cmath>

STDPLSState::STDPLSState(double NewLTPValue, double NewLTDValue): STDPState(NewLTPValue, NewLTDValue){

}

STDPLSState::~STDPLSState() {

}

void STDPLSState::ApplyPresynapticSpike(){
	float PreActivity = this->GetPresynapticActivity();

	// Reset incoming activity
	PreActivity = 1;

	// Store the activity in the state variable
	this->SetStateVariableAt(0,PreActivity);
}

void STDPLSState::ApplyPostsynapticSpike(){
	float PostActivity = this->GetPostsynapticActivity();

	// Reset incoming activity
	PostActivity = 1;

	// Store the activity in the state variable
	this->SetStateVariableAt(1,PostActivity);
}


