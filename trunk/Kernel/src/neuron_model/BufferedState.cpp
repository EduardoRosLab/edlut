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

BufferedState::BufferedState(unsigned int NumVariables, unsigned int NumBuffers):
	NeuronState(NumVariables), FirstElement(0), LastElement(0), BufferAmplitude(0), NumberOfElements(0), NumberOfBuffers(NumBuffers) {
		FirstElement = (ActivityNode **) new ActivityNode * [NumberOfBuffers];
		LastElement = (ActivityNode **) new ActivityNode * [NumberOfBuffers];
		BufferAmplitude = (float *) new float [NumberOfBuffers];
		NumberOfElements = (unsigned int *) new unsigned int [NumberOfBuffers];
}

BufferedState::BufferedState(const BufferedState & OldState): NeuronState(OldState), FirstElement(0), LastElement(0),
		BufferAmplitude(0), NumberOfElements(0), NumberOfBuffers(OldState.NumberOfBuffers) {

	FirstElement = (ActivityNode **) new ActivityNode * [NumberOfBuffers];
	LastElement = (ActivityNode **) new ActivityNode * [NumberOfBuffers];
	BufferAmplitude = (float *) new float [NumberOfBuffers];
	NumberOfElements = (unsigned int *) new unsigned int [NumberOfBuffers];

	for (unsigned int i=0; i<this->NumberOfBuffers; ++i){
		this->BufferAmplitude[i] = OldState.BufferAmplitude[i];

		ActivityNode * Iterator = OldState.FirstElement[i];

		while (Iterator!=0){
			ActivityNode * NewElement = (ActivityNode *) new ActivityNode;

			NewElement->Spike = Iterator->Spike;
			NewElement->NextNode = 0;

			if (this->FirstElement[i]==0){
				// This is the first element of the list
				this->FirstElement[i] = NewElement;
				this->LastElement[i] = NewElement;
			} else {
				// Add the element after the last element
				this->LastElement[i]->NextNode = NewElement;
				this->LastElement[i] = NewElement;
			}

			this->NumberOfElements[i] ++;

			Iterator = Iterator->NextNode;
		}
	}
}

void BufferedState::SetBufferAmplitude(unsigned int NumBuffer, float BufferAmpl){
	this->BufferAmplitude[NumBuffer] = BufferAmpl;
}

BufferedState::~BufferedState() {
	// TODO Auto-generated destructor stub
	for (unsigned int i=0; i<this->NumberOfBuffers; ++i){
		ActivityNode * Iterator = this->FirstElement[i];

		while (Iterator!=0){
			ActivityNode * NextElement = Iterator->NextNode;

			delete Iterator;

			Iterator = NextElement;
		}

	}

	delete [] this->NumberOfElements;
	this->NumberOfElements = 0;
	delete [] this->BufferAmplitude;
	this->BufferAmplitude = 0;
	delete [] this->FirstElement;
	this->FirstElement = 0;
	delete [] this->LastElement;
	this->LastElement = 0;
}

void BufferedState::AddActivity(Interconnection * InputConnection){
	ActivityNode * NewElement = (ActivityNode *) new ActivityNode;

	unsigned int NumBuffer = (unsigned int) InputConnection->GetType();

	NewElement->Spike.first = 0;
	NewElement->Spike.second = InputConnection;
	NewElement->NextNode = 0;

	if (this->FirstElement[NumBuffer]==0){
		// This is the first element of the list
		this->FirstElement[NumBuffer] = NewElement;
		this->LastElement[NumBuffer] = NewElement;
	} else {
		// Add the element after the last element
		this->LastElement[NumBuffer]->NextNode = NewElement;
		this->LastElement[NumBuffer] = NewElement;
	}

	this->NumberOfElements[NumBuffer] ++;
}

void BufferedState::CheckActivity(){
	for (unsigned int i=0; i<this->NumberOfBuffers; ++i){
		// If the first element is older than we accept, remove it.
		ActivityNode * Iterator = this->FirstElement[i];
		while (Iterator!=0 && Iterator->Spike.first>this->BufferAmplitude[i]){
			ActivityNode * Next = Iterator->NextNode;
			delete Iterator;
			this->FirstElement[i] = Next;
			if (Next==0){
				// Empty buffer
				this->LastElement[i] = 0;
			}
			Iterator = Next;
			this->NumberOfElements[i] --;
		}
	}
}

void BufferedState::AddElapsedTime(float ElapsedTime){
	NeuronState::AddElapsedTime(ElapsedTime);

	for (unsigned int i=0; i<this->NumberOfBuffers; ++i){
		ActivityNode * Iterator = this->FirstElement[i];
		while (Iterator!=0){
			Iterator->Spike.first += ElapsedTime;
			Iterator = Iterator->NextNode;
		}
	}

	this->CheckActivity();
}

unsigned int BufferedState::GetNumberOfSpikes(unsigned int NumBuffer){
	return this->NumberOfElements[NumBuffer];
}

double BufferedState::GetSpikeTimeAt(unsigned int Position, unsigned int NumBuffer){
	ActivityNode * Iterator = this->FirstElement[NumBuffer];
	for (unsigned int i = 0; i<Position && Iterator!=0; ++i, Iterator=Iterator->NextNode){
	}
	return (Iterator==0)?-1:Iterator->Spike.first;
}

Interconnection * BufferedState::GetInterconnectionAt(unsigned int Position, unsigned int NumBuffer){
	ActivityNode * Iterator = this->FirstElement[NumBuffer];
	for (unsigned int i = 0; i<Position && Iterator!=0; ++i, Iterator=Iterator->NextNode){
	}
	return (Iterator==0)?0:Iterator->Spike.second;
}

BufferedState::Iterator BufferedState::Begin(unsigned int NumBuffer){
	return Iterator(this->FirstElement[NumBuffer]);
}

BufferedState::Iterator BufferedState::End(){
	return Iterator();
}

BufferedState::Iterator::Iterator():element(0){}

BufferedState::Iterator::Iterator(const BufferedState::Iterator & ItAux){
	this->element = ItAux.element;
}

BufferedState::Iterator::Iterator(ActivityNode * ElemAux){
	this->element=ElemAux;
}

BufferedState::Iterator & BufferedState::Iterator::operator++(){
	this->element = this->element->NextNode;

	return *this;
}

bool BufferedState::Iterator::operator==(BufferedState::Iterator Aux){
	return this->element==Aux.element;
}

bool BufferedState::Iterator::operator!=(BufferedState::Iterator Aux){
	return this->element!=Aux.element;
}

double BufferedState::Iterator::GetSpikeTime(){
	return this->element->Spike.first;
}

Interconnection * BufferedState::Iterator::GetConnection(){
	return this->element->Spike.second;
}
