/***************************************************************************
 *                           VectorSRMState.cpp                            *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@atc.ugr.es         *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/VectorSRMState.h"

VectorSRMState::VectorSRMState(unsigned int NumVariables, unsigned int NumBuffers, bool isTimeDriven): VectorBufferedState(NumVariables, NumBuffers, isTimeDriven){

}

VectorSRMState::~VectorSRMState(){

}

VectorSRMState::VectorSRMState(const VectorSRMState & OldState): VectorBufferedState(OldState){

}

unsigned int VectorSRMState::GetNumberOfPrintableValues(){
	return VectorBufferedState::GetNumberOfPrintableValues()+1;
}

double VectorSRMState::GetPrintableValuesAt(int index, int position){
	if (position<VectorBufferedState::GetNumberOfPrintableValues()){
		return VectorBufferedState::GetPrintableValuesAt(index, position);
	} else if (position==VectorBufferedState::GetNumberOfPrintableValues()) {
		return this->GetLastSpikeTime(index);
	} else return -1;
}


void VectorSRMState::InitializeSRMStates(int size, float * initialization){
	InitializeBufferedStates(size, initialization);
}