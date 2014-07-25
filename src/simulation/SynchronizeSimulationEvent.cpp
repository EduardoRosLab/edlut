/***************************************************************************
 *                           SynchronizeSimulationEvent.cpp                *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/simulation/SynchronizeSimulationEvent.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"
#include "../../include/openmp/openmp.h"

  	
SynchronizeSimulationEvent::SynchronizeSimulationEvent(double NewTime): Event(NewTime){
}
   		
SynchronizeSimulationEvent::~SynchronizeSimulationEvent(){
}

void SynchronizeSimulationEvent::ProcessEvent(Simulation * CurrentSimulation, volatile int * RealTimeRestriction){
	CurrentSimulation->SetSynchronizeSimulationEvent(omp_get_thread_num());
}

void SynchronizeSimulationEvent::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->SetSynchronizeSimulationEvent(omp_get_thread_num());
}


void SynchronizeSimulationEvent::PrintType(){
	cout<<"SynchronizeSimulationEvent"<<endl;
}
   

int SynchronizeSimulationEvent::ProcessingPriority(){
	return 3;
}


