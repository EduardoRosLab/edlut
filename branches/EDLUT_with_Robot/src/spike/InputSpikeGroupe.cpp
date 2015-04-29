/***************************************************************************
 *                           InputSpikeGroupe.cpp                          *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#include "../../include/spike/InputSpikeGroupe.h"

#include "../../include/spike/Neuron.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/Simulation.h"

#include "../../include/neuron_model/NeuronState.h"

#include "../../include/communication/OutputSpikeDriver.h"

#include "../../include/openmp/openmp.h"

#include "../../include/spike/TimeDrivenPropagatedSpike.h"

   	
InputSpikeGroupe::InputSpikeGroupe(double NewTime, Neuron ** NewSources, int NewNElements): Spike(NewTime,NewSources[0]), sources(NewSources), NElements(NewNElements){
}
   		
InputSpikeGroupe::~InputSpikeGroupe(){
	delete sources;
}

//void InputSpikeGroupe::ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction){
//
//	if(RealTimeRestriction<2){
//
//
//		for(int j=0; j<this->NElements; j++){
//			this->SetSource(sources[j]);
//			CurrentSimulation->WriteSpike(this);
//			
//			for(int i=0; i<NumberOfOpenMPQueues; i++){
//				if (source->IsOutputConnected(i)){
//					PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + source->GetOutputConnectionAt(i,0)->GetDelay(), source, 0,i);
//					if(i==omp_get_thread_num()){
//						CurrentSimulation->GetQueue()->InsertEvent(i,spike);
//					}else{
//						CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
//					}
//
//				}
//			}
//		}
//	}
//}
//
//void InputSpikeGroupe::ProcessEvent(Simulation * CurrentSimulation){
//	for(int j=0; j<this->NElements; j++){
//		this->SetSource(sources[j]);
//		CurrentSimulation->WriteSpike(this);
//	
//		for(int i=0; i<NumberOfOpenMPQueues; i++){
//			if (source->IsOutputConnected(i)){
//				PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + source->GetOutputConnectionAt(i,0)->GetDelay(), source, 0,i);
//				if(i==omp_get_thread_num()){
//					CurrentSimulation->GetQueue()->InsertEvent(i,spike);
//				}else{
//					CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
//				}
//
//			}
//		}
//	}
//}

//void InputSpikeGroupe::ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction){
//
//	if(RealTimeRestriction<2){
//
//		/////////////////revisar////////////////
//		TimeDrivenPropagatedSpike** timeDrivenPropagatedSpike=(TimeDrivenPropagatedSpike**) new TimeDrivenPropagatedSpike * [NumberOfOpenMPQueues];
//		for(int i=0; i<NumberOfOpenMPQueues; i++){
//			timeDrivenPropagatedSpike[i]=new TimeDrivenPropagatedSpike(/*revisar*/this->GetTime()+0.001,i,this->NElements);
//		}
//
//
//		for(int j=0; j<this->NElements; j++){
//			this->SetSource(sources[j]);
//			CurrentSimulation->WriteSpike(this);
//			
//			for(int i=0; i<NumberOfOpenMPQueues; i++){
//				if (source->IsOutputConnected(i)){
//					timeDrivenPropagatedSpike[i]->IncludeNewSource(/*revisar*/source->GetOutputNumber(i),/*revisar*/source->GetOutputConnectionAt(i, 0));
//				}
//			}
//		}
//
//		for(int i=0; i<NumberOfOpenMPQueues; i++){
//			if (timeDrivenPropagatedSpike[i]->GetN_Elementes()>0){
//				if(i==omp_get_thread_num()){
//					CurrentSimulation->GetQueue()->InsertEvent(i,timeDrivenPropagatedSpike[i]);
//				}else{
//					CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,timeDrivenPropagatedSpike[i]);
//				}
//			}else{
//				delete timeDrivenPropagatedSpike[i];
//			}
//		}
//		delete timeDrivenPropagatedSpike;
//	}
//}

void InputSpikeGroupe::ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction){

	if(RealTimeRestriction<2){
		for(int j=0; j<this->NElements; j++){
			this->SetSource(sources[j]);
			CurrentSimulation->WriteSpike(this);
			
			// Generate the output activity
			for(int i=0; i<NumberOfOpenMPQueues; i++){
				if (sources[j]->IsOutputConnected(i)){
					PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + sources[j]->PropagationStructure->SynapseDelay[i][0], sources[j], 0,sources[j]->PropagationStructure->NDifferentDelays[i], i);
					if(i==sources[j]->get_OpenMP_queue_index()){
						CurrentSimulation->GetQueue()->InsertEvent(i,spike);
					}else{
						CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
					}
				}
			}
		}
	}
}



//void InputSpikeGroupe::ProcessEvent(Simulation * CurrentSimulation){
//	/////////////////revisar////////////////
//	TimeDrivenPropagatedSpike** timeDrivenPropagatedSpike=(TimeDrivenPropagatedSpike**) new TimeDrivenPropagatedSpike * [NumberOfOpenMPQueues];
//	for(int i=0; i<NumberOfOpenMPQueues; i++){
//		timeDrivenPropagatedSpike[i]=new TimeDrivenPropagatedSpike(/*revisar*/this->GetTime()+0.001,i,this->NElements);
//	}
//
//
//	for(int j=0; j<this->NElements; j++){
//		this->SetSource(sources[j]);
//		CurrentSimulation->WriteSpike(this);
//		
//		for(int i=0; i<NumberOfOpenMPQueues; i++){
//			if (source->IsOutputConnected(i)){
//				timeDrivenPropagatedSpike[i]->IncludeNewSource(/*revisar*/source->GetOutputNumber(i),/*revisar*/source->GetOutputConnectionAt(i, 0));
//			}
//		}
//	}
//
//	for(int i=0; i<NumberOfOpenMPQueues; i++){
//		if (timeDrivenPropagatedSpike[i]->GetN_Elementes()>0){
//			if(i==omp_get_thread_num()){
//				CurrentSimulation->GetQueue()->InsertEvent(i,timeDrivenPropagatedSpike[i]);
//			}else{
//				CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,timeDrivenPropagatedSpike[i]);
//			}
//		}else{
//			delete timeDrivenPropagatedSpike[i];
//		}
//	}
//	delete timeDrivenPropagatedSpike;
//}

void InputSpikeGroupe::ProcessEvent(Simulation * CurrentSimulation){


	for(int j=0; j<this->NElements; j++){
		this->SetSource(sources[j]);
		CurrentSimulation->WriteSpike(this);
		
		// Generate the output activity
		for(int i=0; i<NumberOfOpenMPQueues; i++){
			if (sources[j]->IsOutputConnected(i)){
				PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + sources[j]->PropagationStructure->SynapseDelay[i][0], sources[j], 0, sources[j]->PropagationStructure->NDifferentDelays[i],  i);
				if(i==sources[j]->get_OpenMP_queue_index()){
					CurrentSimulation->GetQueue()->InsertEvent(i,spike);
				}else{
					CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
				}
			}
		}
	}
}


void InputSpikeGroupe::PrintType(){
	cout<<"InputSpikeGroupe"<<endl;
}

   	
