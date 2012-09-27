/***************************************************************************
 *                           EventQueueSTL.cpp                                *
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

#include "../../include/simulation/EventQueueSTL.h"

#include "../../include/simulation/Event.h"

EventQueueSTL::EventQueueSTL() : Events(1,(Event *)0){

}
   		
EventQueueSTL::~EventQueueSTL(){
	for (unsigned int i=0; i<Events.size(); ++i){
		delete Events[i];
	}
}

inline void EventQueueSTL::SwapEvents(unsigned int c1, unsigned int c2){
	Event * exchange;
	exchange=Events[c2];
	Events[c2]=Events[c1];
	Events[c1]=exchange;
}
   		
void EventQueueSTL::InsertEvent(Event * event){
	Events.push_back(event);
      
  	for(unsigned int c=Size();c>1 && Events[c/2]->GetTime() > Events[c]->GetTime(); c/=2){
    	SwapEvents(c, c/2);
  	}
    
    return;
}

unsigned int EventQueueSTL::Size() const{
	return (Events.size()-1);
}
   		
Event * EventQueueSTL::RemoveEvent(void){
	unsigned int c,p;
   	
   	Event * first = 0;
   	if(Size()>0){
   		first=Events[1];
      
      	//Events[1]=Events.back();
		Events[1]=Events[this->Size()];
      	Events.pop_back();
      
      	p=1;
      	for(c=p*2;c<Size();p=c,c=p*2){
      		if(Events[c]->GetTime() > Events[c+1]->GetTime())
      			c++;
      		
      		if(Events[c]->GetTime() < Events[p]->GetTime())
            	SwapEvents(p, c);
         	else
            	break;
        }
      
      	if(c==Size() && Events[p]->GetTime() > Events[c]->GetTime())
        	SwapEvents(p, c);
	}
    
    return(first);
}
   		
double EventQueueSTL::FirstEventTime() const{
	double ti;
	
	if(Size()>0)
		ti=Events[1]->GetTime();
   	else
    	ti=-1.0;
   
   	return(ti);		
}
