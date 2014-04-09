/***************************************************************************
 *                           StopSimulationEvent.cpp                        *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido                        *
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

#include "../../include/simulation/StopSimulationEvent.h"

#include "../../include/simulation/Simulation.h"

StopSimulationEvent::StopSimulationEvent():Event(0){
}
   	
StopSimulationEvent::StopSimulationEvent(double NewTime): Event(NewTime){
}
   		
StopSimulationEvent::~StopSimulationEvent(){
}

void StopSimulationEvent::ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction){
	CurrentSimulation->StopSimulation();		
}
   	


