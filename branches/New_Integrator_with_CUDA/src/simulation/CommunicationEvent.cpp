/***************************************************************************
 *                           CommunicationEvent.cpp                        *
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

#include "../../include/simulation/CommunicationEvent.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

CommunicationEvent::CommunicationEvent():Event(0){
}
   	
CommunicationEvent::CommunicationEvent(double NewTime): Event(NewTime){
}
   		
CommunicationEvent::~CommunicationEvent(){
}

void CommunicationEvent::ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction){
	
	// Send the outputs
	CurrentSimulation->SendOutput();
	
	// Get the inputs
	CurrentSimulation->GetInput();
	
	if (CurrentSimulation->GetSimulationStep()>0.0){
		CommunicationEvent * NewEvent = new CommunicationEvent(this->GetTime()+CurrentSimulation->GetSimulationStep());
		CurrentSimulation->GetQueue()->InsertEvent(NewEvent);
	}		
}
   	

