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

SRMState::SRMState(unsigned int NumVariables, float BufferAmpl, unsigned int MaxSize): BufferedState(NumVariables,BufferAmpl,MaxSize), LastSpikeTime(100){

}

SRMState::~SRMState(){

}

SRMState::SRMState(const SRMState & OldState): BufferedState(OldState), LastSpikeTime(OldState.LastSpikeTime){

}

void SRMState::AddElapsedTime(float ElapsedTime){
	this->LastSpikeTime += ElapsedTime;

	BufferedState::AddElapsedTime(ElapsedTime);
}

void SRMState::NewFiredSpike(){
	this->LastSpikeTime = 0;
}

double SRMState::GetLastSpikeTime(){
	return this->LastSpikeTime;
}
