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

#include "../../include/simulation/ExponentialTable.h"

#include <cmath>

STDPLSState::STDPLSState(unsigned int NumSynapses, double NewLTPValue, double NewLTDValue): STDPState(NumSynapses, NewLTPValue, NewLTDValue){

}

STDPLSState::~STDPLSState() {

}

//void STDPLSState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
//	if(pre_post){
//		float PreActivity = this->GetPresynapticActivity(index);
//
//		float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));
//
//		// Accumulate activity since the last update time
//		PreActivity *= exp(-ElapsedTime*this->inv_LTPTau);
//
//
//		// Store the activity in state variables
//		this->SetStateVariableAt(index, 0, PreActivity);
//	}else{
//		float PostActivity = this->GetPostsynapticActivity(index);
//
//		float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));
//
//		// Accumulate activity since the last update time
//		PostActivity *= exp(-ElapsedTime*this->inv_LTDTau);
//
//		// Store the activity in state variables
//		this->SetStateVariableAt(index, 1, PostActivity);
//	}
//
//	this->SetLastUpdateTime(index, NewTime);
//}

void STDPLSState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
	float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));
	if(pre_post){
		//Accumulate activity since the last update time
		this->multiplyStateVaraibleAt(index,0,ExponentialTable::GetResult(-ElapsedTime*this->inv_LTPTau));
	}else{
		//Accumulate activity since the last update time
		this->multiplyStateVaraibleAt(index,1,ExponentialTable::GetResult(-ElapsedTime*this->inv_LTDTau));
	}
	this->SetLastUpdateTime(index, NewTime);
}



void STDPLSState::ApplyPresynapticSpike(unsigned int index){
	// Store the activity in the state variable
	this->SetStateVariableAt(index, 0, 1.0f);
}

void STDPLSState::ApplyPostsynapticSpike(unsigned int index){
	// Store the activity in the state variable
	this->SetStateVariableAt(index, 1, 1.0f);
}


