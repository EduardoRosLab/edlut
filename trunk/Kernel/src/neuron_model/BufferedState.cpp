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

BufferedState::BufferedState(unsigned int NumVariables, float BufferAmpl):
	NeuronState(NumVariables), FirstElement(0), LastElement(0), BufferAmplitude(BufferAmpl), NumberOfElements(0) {
}

BufferedState::BufferedState(const BufferedState & OldState): NeuronState(OldState), FirstElement(0), LastElement(0),
		BufferAmplitude(OldState.BufferAmplitude), NumberOfElements(0) {
	ActivityNode * Iterator = OldState.FirstElement;

	while (Iterator!=0){
		ActivityNode * NewElement = (ActivityNode *) new ActivityNode;

		NewElement->Activity = Iterator->Activity;
		NewElement->NextNode = 0;

		if (this->FirstElement==0){
			// This is the first element of the list
			this->FirstElement = NewElement;
			this->LastElement = NewElement;
		} else {
			// Add the element after the last element
			this->LastElement->NextNode = NewElement;
			this->LastElement = NewElement;
		}

		this->NumberOfElements ++;

		Iterator = Iterator->NextNode;
	}
}

BufferedState::~BufferedState() {
	// TODO Auto-generated destructor stub
	ActivityNode * Iterator = this->FirstElement;

	while (Iterator!=0){
		ActivityNode * NextElement = Iterator->NextNode;

		delete Iterator;

		Iterator = NextElement;
	}

	this->NumberOfElements = 0;
	this->FirstElement = 0;
	this->LastElement = 0;
}

void BufferedState::AddActivity(Interconnection * InputConnection){
	ActivityNode * NewElement = (ActivityNode *) new ActivityNode;

	NewElement->Activity.first = 0;
	NewElement->Activity.second = InputConnection;
	NewElement->NextNode = 0;

	if (this->FirstElement==0){
		// This is the first element of the list
		this->FirstElement = NewElement;
		this->LastElement = NewElement;
	} else {
		// Add the element after the last element
		this->LastElement->NextNode = NewElement;
		this->LastElement = NewElement;
	}

	this->NumberOfElements ++;
}

void BufferedState::CheckActivity(){
	// If the first element is older than we accept, remove it.
	ActivityNode * Iterator = this->FirstElement;
	while (Iterator!=0 && Iterator->Activity.first>this->BufferAmplitude){
		ActivityNode * Next = Iterator->NextNode;
		delete Iterator;
		this->FirstElement = Next;
		if (Next==0){
			// Empty buffer
			this->LastElement = 0;
		}
		Iterator = Next;
		this->NumberOfElements --;
	}
}

void BufferedState::AddElapsedTime(float ElapsedTime){
	NeuronState::AddElapsedTime(ElapsedTime);

	ActivityNode * Iterator = this->FirstElement;
	while (Iterator!=0){
		Iterator->Activity.first += ElapsedTime;
		Iterator = Iterator->NextNode;
	}

	this->CheckActivity();
}

unsigned int BufferedState::GetNumberOfSpikes(){
	return this->NumberOfElements;
}

double BufferedState::GetSpikeTimeAt(unsigned int Position){
	ActivityNode * Iterator = this->FirstElement;
	for (unsigned int i = 0; i<Position && Iterator!=0; ++i, Iterator=Iterator->NextNode){
	}
	return (Iterator==0)?-1:Iterator->Activity.first;
}

Interconnection * BufferedState::GetInterconnectionAt(unsigned int Position){
	ActivityNode * Iterator = this->FirstElement;
	for (unsigned int i = 0; i<Position && Iterator!=0; ++i, Iterator=Iterator->NextNode){
	}
	return (Iterator==0)?0:Iterator->Activity.second;
}

BufferedState::Iterator BufferedState::Begin(){
	return Iterator(this->FirstElement);
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
	return this->element->Activity.first;
}

Interconnection * BufferedState::Iterator::GetConnection(){
	return this->element->Activity.second;
}
