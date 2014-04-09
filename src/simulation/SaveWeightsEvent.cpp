/***************************************************************************
 *                           SaveWeightsEvent.cpp                          *
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

#include "../../include/simulation/SaveWeightsEvent.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

SaveWeightsEvent::SaveWeightsEvent():Event(0){
}
   	
SaveWeightsEvent::SaveWeightsEvent(double NewTime): Event(NewTime){
}
   		
SaveWeightsEvent::~SaveWeightsEvent(){
}

void SaveWeightsEvent::ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction){
	CurrentSimulation->SaveWeights();
	if (CurrentSimulation->GetSaveStep()>0.0){
		SaveWeightsEvent * NewEvent = new SaveWeightsEvent(this->GetTime()+CurrentSimulation->GetSaveStep());
		CurrentSimulation->GetQueue()->InsertEvent(NewEvent);
	}		
}
   	

