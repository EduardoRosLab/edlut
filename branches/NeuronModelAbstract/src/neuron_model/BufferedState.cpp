/***************************************************************************
 *                           BufferedState.cpp                             *
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

#include "../../include/neuron_model/BufferedState.h"

BufferedState::BufferedState(unsigned int NumVariables, float BufferAmpl, unsigned int MaxSize):
	NeuronState(NumVariables), MaximumSize(MaxSize), FirstIndex(0), LastIndex(0){
	// TODO Auto-generated constructor stub
	this->ActivityBuffer = (InputActivity *) new InputActivity [this->MaximumSize+1];
}

BufferedState::BufferedState(const BufferedState & OldState): NeuronState(OldState), BufferAmplitude(OldState.BufferAmplitude),
		MaximumSize(OldState.MaximumSize), FirstIndex(OldState.FirstIndex), LastIndex(OldState.LastIndex) {
	this->ActivityBuffer = (InputActivity *) new InputActivity [this->MaximumSize+1];

	for (unsigned int i=0; i<this->MaximumSize+1; ++i){
		this->ActivityBuffer[i] = OldState.ActivityBuffer[i];
	}
}

BufferedState::~BufferedState() {
	// TODO Auto-generated destructor stub
}

void BufferedState::AddActivity(Interconnection * InputConnection){
	this->FirstIndex = (this->FirstIndex+1)%(this->MaximumSize+1);
	this->ActivityBuffer[this->FirstIndex].first = 0;
	this->ActivityBuffer[this->FirstIndex].second = InputConnection;

	// If the buffer is full, remove the last element
	if (this->FirstIndex==this->LastIndex){
		this->LastIndex = (this->LastIndex+1)%(this->MaximumSize+1);
	}
}

void BufferedState::CheckActivity(){
	// If the last element is older than we accept, remove it.
	unsigned int index = (this->LastIndex+1)%(this->MaximumSize+1);
	while (this->FirstIndex!=this->LastIndex && this->ActivityBuffer[index].first>this->BufferAmplitude){
		this->LastIndex = index;
		index = (index+1)%(this->MaximumSize+1);
	}
}

void BufferedState::AddElapsedTime(float ElapsedTime){
	NeuronState::AddElapsedTime(ElapsedTime);

	int index = this->FirstIndex;
	while (index!=this->LastIndex){
		this->ActivityBuffer[index].first += ElapsedTime;
		index -= 1;
		index = (index<0) ? this->MaximumSize : index;
		//index = (index-1)%(this->MaximumSize+1);
	}
}

unsigned int BufferedState::GetNumberOfSpikes(){
	return (this->FirstIndex-this->LastIndex)%(this->MaximumSize+1);
}

double BufferedState::GetSpikeTimeAt(unsigned int Position){
	return this->ActivityBuffer[(this->FirstIndex-Position)%(this->MaximumSize+1)].first;
}

Interconnection * BufferedState::GetInterconnectionAt(unsigned int Position){
	return this->ActivityBuffer[(this->FirstIndex-Position)%(this->MaximumSize+1)].second;
}
