/***************************************************************************
 *                           Event.cpp                                     *
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

#include "../../include/simulation/Event.h"

#include "../../include/simulation/Simulation.h"

Event::Event():time(0){
}
   	
Event::Event(double NewTime): time(NewTime){
}
   		
Event::~Event(){
}
   	
double Event::GetTime() const{
	return time;
}
   		
void Event::SetTime (double NewTime){
	time = NewTime;
}


