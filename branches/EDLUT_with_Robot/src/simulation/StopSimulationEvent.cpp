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
#include "../../include/openmp/openmp.h"

StopSimulationEvent::StopSimulationEvent():Event(0){
}
   	
StopSimulationEvent::StopSimulationEvent(double NewTime): Event(NewTime){
}
   		
StopSimulationEvent::~StopSimulationEvent(){
}

void StopSimulationEvent::ProcessEvent(Simulation * CurrentSimulation, volatile int * RealTimeRestriction){
	CurrentSimulation->StopSimulation(omp_get_thread_num());	/*asdfgf*/	
}

void StopSimulationEvent::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->StopSimulation(omp_get_thread_num());	/*asdfgf*/	
}

void StopSimulationEvent::PrintType(){
	cout<<"StopSimulationEvent"<<endl;
}
   	
int StopSimulationEvent::ProcessingPriority(){
	return 1;
}


