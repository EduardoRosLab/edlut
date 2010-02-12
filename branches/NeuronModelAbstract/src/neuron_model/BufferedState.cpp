/*
 * BufferedState.cpp
 *
 *  Created on: 09/02/2010
 *      Author: jesus
 */

#include "../../include/neuron_model/BufferedState.h"

BufferedState::BufferedState(unsigned int NumVariables, float BufferAmpl, unsigned int MaxSize):
	NeuronState(NumVariables), MaximumSize(MaxSize), FirstIndex(0), LastIndex(0){
	// TODO Auto-generated constructor stub
	this->ActivityBuffer = (InputActivity *) new InputActivity [this->MaximumSize+1];

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
		index = (this->LastIndex+1)%(this->MaximumSize+1);
	}
}

void BufferedState::AddElapsedTime(float ElapsedTime){
	unsigned int index = (this->LastIndex+1)%(this->MaximumSize+1);
	while (index!=this->LastIndex){
		this->ActivityBuffer[index].first += ElapsedTime;
		index = (this->LastIndex+1)%(this->MaximumSize+1);
	}
}

unsigned int BufferedState::GetNumberOfSpikes(){
	return (this->FirstIndex-this->LastIndex)%(this->MaximumSize+1);
}

double BufferedState::GetSpikeTimeAt(unsigned int Position){
	return this->ActivityBuffer[(this->LastIndex+Position)%(this->MaximumSize+1)].first;
}

Interconnection * BufferedState::GetInterconnectionAt(unsigned int Position){
	return this->ActivityBuffer[(this->LastIndex+Position)%(this->MaximumSize+1)].second;
}
