/***************************************************************************
 *                           SynchronizeActivityEvent.cpp                  *
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

#include "../../include/simulation/SynchronizeActivityEvent.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/simulation/SynchronizeSimulationEvent.h"

#include "../../include/openmp/openmp.h"

   	
SynchronizeActivityEvent::SynchronizeActivityEvent(double NewTime, Simulation * CurrentSimulation): Event(NewTime){
	for(int i=0; i<CurrentSimulation->GetNumberOfQueues(); i++){
		SynchronizeSimulationEvent * NewEvent = new SynchronizeSimulationEvent(NewTime);
		CurrentSimulation->GetQueue()->InsertEvent(i,NewEvent);
	}
}
   		
SynchronizeActivityEvent::~SynchronizeActivityEvent(){
}

void SynchronizeActivityEvent::ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction){
	if(RealTimeRestriction<2){
		for(int i=0; i<CurrentSimulation->GetNumberOfQueues(); i++){
			#ifdef _OPENMP 
				#if	_OPENMP >= OPENMPVERSION30
					#pragma omp task firstprivate (i)
				#endif
			#endif
			{
				CurrentSimulation->GetQueue()->InsertBufferInQueue(i);
			}
		}
		#ifdef _OPENMP 
			#if	_OPENMP >= OPENMPVERSION30
				#pragma omp taskwait 
			#endif
		#endif
	}else{
		for(int i=0; i<CurrentSimulation->GetNumberOfQueues(); i++){
			#ifdef _OPENMP 
				#if	_OPENMP >= OPENMPVERSION30
					#pragma omp task firstprivate (i)
				#endif
			#endif
			{
				CurrentSimulation->GetQueue()->ResetBuffer(i);
			}
		}
		#ifdef _OPENMP 
			#if	_OPENMP >= OPENMPVERSION30
				#pragma omp taskwait 
			#endif
		#endif
	}


	if (CurrentSimulation->GetMinInterpropagationTime()>0.0){
		SynchronizeActivityEvent * NewEvent = new SynchronizeActivityEvent(this->GetTime()+CurrentSimulation->GetMinInterpropagationTime(), CurrentSimulation);
		CurrentSimulation->GetQueue()->InsertEventWithSynchronization(NewEvent);
	}
}

void SynchronizeActivityEvent::ProcessEvent(Simulation * CurrentSimulation){
	for(int i=0; i<CurrentSimulation->GetNumberOfQueues(); i++){
		#ifdef _OPENMP 
			#if	_OPENMP >= OPENMPVERSION30
				#pragma omp task firstprivate (i)
			#endif
		#endif
		{
			CurrentSimulation->GetQueue()->InsertBufferInQueue(i);
		}
	}
	#ifdef _OPENMP 
		#if	_OPENMP >= OPENMPVERSION30
			#pragma omp taskwait 
		#endif
	#endif

	if (CurrentSimulation->GetMinInterpropagationTime()>0.0){
		SynchronizeActivityEvent * NewEvent = new SynchronizeActivityEvent(this->GetTime()+CurrentSimulation->GetMinInterpropagationTime(), CurrentSimulation);
		CurrentSimulation->GetQueue()->InsertEventWithSynchronization(NewEvent);
	}
}
   	
void SynchronizeActivityEvent::PrintType(){
	cout<<"SynchronizeActivityEvent"<<endl;
}

enum EventPriority SynchronizeActivityEvent::ProcessingPriority(){
	return SYNCHRONIZEACTIVITYEVENT;
}

