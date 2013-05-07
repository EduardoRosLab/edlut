/***************************************************************************
 *                           EndSimulationEvent.cpp                        *
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

#include "../../include/simulation/EndSimulationEvent.h"

#include "../../include/simulation/Simulation.h"

EndSimulationEvent::EndSimulationEvent():Event(0){
}
   	
EndSimulationEvent::EndSimulationEvent(double NewTime): Event(NewTime){
}
   		
EndSimulationEvent::~EndSimulationEvent(){
}

void EndSimulationEvent::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->EndSimulation();		
}
   	


