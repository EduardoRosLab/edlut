/***************************************************************************
 *                           InputSpike.cpp                                *
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

#include "../../include/spike/InputSpike.h"

#include "../../include/spike/Neuron.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/Simulation.h"

#include "../../include/neuron_model/NeuronState.h"

#include "../../include/communication/OutputSpikeDriver.h"

#include "../../include/openmp/openmp.h"


InputSpike::InputSpike():Spike() {
}
   	
InputSpike::InputSpike(double NewTime, Neuron * NewSource): Spike(NewTime,NewSource){
}
   		
InputSpike::~InputSpike(){
}

void InputSpike::ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction){

	if(RealTimeRestriction<2){
		
		CurrentSimulation->WriteSpike(this);
		
		for(int i=0; i<NumberOfOpenMPQueues; i++){
			if (source->IsOutputConnected(i)){
				PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + source->GetOutputConnectionAt(i,0)->GetDelay(), source, 0, source->PropagationStructure->NDifferentDelays[i], i);
				if(i==omp_get_thread_num()){
					CurrentSimulation->GetQueue()->InsertEvent(i,spike);
				}else{
					CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
				}

			}
		}
	}
}

void InputSpike::ProcessEvent(Simulation * CurrentSimulation){

		
	CurrentSimulation->WriteSpike(this);
	
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		if (source->IsOutputConnected(i)){
			PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + source->GetOutputConnectionAt(i,0)->GetDelay(), source, 0, source->PropagationStructure->NDifferentDelays[i], i);
			if(i==omp_get_thread_num()){
				CurrentSimulation->GetQueue()->InsertEvent(i,spike);
			}else{
				CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
			}

		}
	}
}

void InputSpike::PrintType(){
	cout<<"InputSpike"<<endl;
}

   	
