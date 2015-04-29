/***************************************************************************
 *                           TimeDrivenPropagatedSpike.cpp                 *
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

#include "../../include/spike/TimeDrivenPropagatedSpike.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"

#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/learning_rules/LearningRule.h"
#include "../../include/learning_rules/AdditiveKernelChange.h"

#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"


TimeDrivenPropagatedSpike::TimeDrivenPropagatedSpike(double NewTime, int NewOpenMP_index, int NewMaxSize):Spike(NewTime, NULL), OpenMP_index(NewOpenMP_index), MaxSize(NewMaxSize), N_Elements(0) {
	N_ConnectionsWithEqualDelay=new int[NewMaxSize];
	ConnectionsWithEqualDelay=(Interconnection **) new Interconnection * [NewMaxSize];
}
   	
	
TimeDrivenPropagatedSpike::~TimeDrivenPropagatedSpike(){
	delete N_ConnectionsWithEqualDelay;
	delete ConnectionsWithEqualDelay;
}

int TimeDrivenPropagatedSpike::GetN_Elementes(){
	return N_Elements;
}

int TimeDrivenPropagatedSpike::GetMaxSize(){
	return MaxSize;
}

bool TimeDrivenPropagatedSpike::IncludeNewSource(int NewN_connections, Interconnection * NewConnections){
	N_ConnectionsWithEqualDelay[N_Elements]=NewN_connections;
	ConnectionsWithEqualDelay[N_Elements]=NewConnections;

	N_Elements++;
	if(N_Elements<MaxSize){
		return false;
	}
	return true;
}



   	

//Optimized function. This function propagates all neuron output spikes that have the same delay.
void TimeDrivenPropagatedSpike::ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction){

	if(RealTimeRestriction<2){
		CurrentSimulation->IncrementTotalPropagateEventCounter(omp_get_thread_num()); 
		Interconnection * inter;
		for(int i=0; i<N_Elements; i++){
			inter=ConnectionsWithEqualDelay[i];
			CurrentSimulation->IncrementTotalPropagateCounter(omp_get_thread_num(), N_ConnectionsWithEqualDelay[i]);
			for (int j=0; j<N_ConnectionsWithEqualDelay[i]; j++){
				Neuron * TargetNeuron = inter->GetTarget();  // target of the spike

				InternalSpike * Generated = TargetNeuron->GetNeuronModel()->ProcessInputSpike(inter, TargetNeuron, this->time);

				if (Generated!=0){
					CurrentSimulation->GetQueue()->InsertEvent(OpenMP_index,Generated);
				}

				if(RealTimeRestriction<1){
					LearningRule * ConnectionRule = inter->GetWeightChange_withoutPost();
					// If learning, change weights
					if(ConnectionRule != 0){
						ConnectionRule->ApplyPreSynapticSpike(inter,this->time);
					}
					ConnectionRule = inter->GetWeightChange_withPost();
					// If learning, change weights
					if(ConnectionRule != 0){
						ConnectionRule->ApplyPreSynapticSpike(inter,this->time);
					}
				}
				inter++;
			}
		}
	}
}


//Optimized function. This function propagates all neuron output spikes that have the same delay.
void TimeDrivenPropagatedSpike::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->IncrementTotalPropagateEventCounter(omp_get_thread_num()); 
	Interconnection * inter;
	for(int i=0; i<N_Elements; i++){
		inter=ConnectionsWithEqualDelay[i];
		CurrentSimulation->IncrementTotalPropagateCounter(omp_get_thread_num(), N_ConnectionsWithEqualDelay[i]);
		for (int j=0; j<N_ConnectionsWithEqualDelay[i]; j++){
			Neuron * TargetNeuron = inter->GetTarget();  // target of the spike

			InternalSpike * Generated = TargetNeuron->GetNeuronModel()->ProcessInputSpike(inter, TargetNeuron, this->time);

			if (Generated!=0){
				CurrentSimulation->GetQueue()->InsertEvent(OpenMP_index,Generated);
			}

			LearningRule * ConnectionRule = inter->GetWeightChange_withoutPost();
			// If learning, change weights
			if(ConnectionRule != 0){
				ConnectionRule->ApplyPreSynapticSpike(inter,this->time);
			}
			ConnectionRule = inter->GetWeightChange_withPost();
			// If learning, change weights
			if(ConnectionRule != 0){
				ConnectionRule->ApplyPreSynapticSpike(inter,this->time);
			}
			inter++;
		}
	}
}




int TimeDrivenPropagatedSpike::GetOpenMP_index() const{
	return OpenMP_index;
}

void TimeDrivenPropagatedSpike::PrintType(){
	cout<<"TimeDrivenPropagatedSpike"<<endl;
}

