/***************************************************************************
 *                           SRMState.cpp                                  *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#include "../../include/neuron_model/SRMState.h"

SRMState::SRMState(unsigned int NumVariables, float BufferAmpl, unsigned int MaxSize): BufferedState(NumVariables,BufferAmpl,MaxSize){

}

SRMState::~SRMState(){

}

SRMState::SRMState(const SRMState & OldState): BufferedState(OldState){

}

unsigned int SRMState::GetNumberOfPrintableValues(){
	return BufferedState::GetNumberOfPrintableValues()+1;
}

double SRMState::GetPrintableValuesAt(unsigned int position){
	if (position<BufferedState::GetNumberOfPrintableValues()){
		return BufferedState::GetStateVariableAt(position);
	} else if (position==BufferedState::GetNumberOfPrintableValues()) {
		return this->GetLastSpikeTime();
	} else return -1;
}
