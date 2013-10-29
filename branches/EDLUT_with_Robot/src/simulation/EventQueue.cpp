/***************************************************************************
 *                           EventQueue.cpp                                *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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

#include "../../include/simulation/EventQueue.h"

#include "../../include/simulation/Event.h"

EventQueue::EventQueue() : Events(0), NumberOfElements(0), AllocatedSize(0) {
	// Allocate memory for a MIN_SIZE sized array
	this->Events = (EventForQueue *) new EventForQueue [MIN_SIZE];

	this->AllocatedSize = MIN_SIZE;
	
	// The first element in the array is discard
	NumberOfElements = 1;

}
   		
EventQueue::~EventQueue(){
	for (unsigned int i=1; i<this->NumberOfElements; ++i){
		delete (this->Events+i)->EventPtr;
	}

	delete [] this->Events;
}

void EventQueue::SwapEvents(unsigned int c1, unsigned int c2){
	EventForQueue exchange;
	exchange=*(this->Events+c2);
	*(this->Events+c2)=*(this->Events+c1);
	*(this->Events+c1)=exchange;
}


void EventQueue::Resize(unsigned int NewSize){
	EventForQueue * Temp = this->Events;

	// Allocate the new array
	this->Events = (EventForQueue *) new EventForQueue [NewSize];

	this->AllocatedSize = NewSize;

	// Copy all the elements from the original array
	for (unsigned int i = 0; i<this->NumberOfElements; ++i){
		*(this->Events+i) = *(Temp+i);
	}
	
	// Release old memory
	delete [] Temp;
}
   		
void EventQueue::InsertEvent(Event * event){

	if (this->NumberOfElements == this->AllocatedSize){
		this->Resize(this->AllocatedSize*RESIZE_FACTOR);
	}
	
	(this->Events+this->NumberOfElements)->EventPtr = event;
	(this->Events+this->NumberOfElements)->Time = event->GetTime();
	
	this->NumberOfElements++;


	for(unsigned int c=this->Size();c>1 && (this->Events+c/2)->Time > (this->Events+c)->Time; c/=2){
    	SwapEvents(c, c/2);
  	}
    
    return;
}

unsigned int EventQueue::Size() const{
	return this->NumberOfElements-1;
}
   		
Event * EventQueue::RemoveEvent(void){
	unsigned int c,p;
double time_c0, time_c1, time_p;
   	
   	Event * first = 0;
	if(this->NumberOfElements>2){
		first=(this->Events+1)->EventPtr;
      
      	//Events[1]=Events.back();
		*(this->Events+1)=*(this->Events+this->Size());
		this->NumberOfElements--;

		if (this->NumberOfElements>MIN_SIZE && this->NumberOfElements<this->AllocatedSize/(RESIZE_FACTOR*2)){
			this->Resize(this->AllocatedSize/RESIZE_FACTOR);
		}
      	
      	p=1;
		for(c=p*2;c<this->Size();p=c,c=p*2){
      		if((this->Events+c)->Time > (this->Events+c+1)->Time)
      			c++;
      		
      		if((this->Events+c)->Time < (this->Events+p)->Time)
            	SwapEvents(p, c);
         	else
            	break;
        }

		if(c==this->Size() && (this->Events+p)->Time > (this->Events+c)->Time)
        	SwapEvents(p, c);
	} else if (this->NumberOfElements==2){
		first = (this->Events+1)->EventPtr;

		this->NumberOfElements--;
	}
    
    return(first);
}
   		
double EventQueue::FirstEventTime() const{
	double ti;
	
	if(this->NumberOfElements>1)
		ti=(this->Events+1)->Time;
   	else
    	ti=-1.0;
   
   	return(ti);		
}

void EventQueue::RemoveSpikes(){
	unsigned int OldNumberOfElements=this->NumberOfElements;
	Event * TmpEvent;

	this->NumberOfElements=1; // Initially resize occupied size of the heap so that all the events are out
	// Reinsert in the heap only the events which are spikes 
	for (unsigned int i = 1; i<OldNumberOfElements; ++i){
		TmpEvent=(this->Events+i)->EventPtr;
		if(!TmpEvent->IsSpike()){
			InsertEvent(TmpEvent);
		}else{
			delete TmpEvent;
		}		
	}
}
