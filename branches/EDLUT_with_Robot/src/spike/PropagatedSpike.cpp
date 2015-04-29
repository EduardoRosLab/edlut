/***************************************************************************
 *                           PropagatedSpike.cpp                           *
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

#include "../../include/spike/PropagatedSpike.h"
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


PropagatedSpike::PropagatedSpike(int NewOpenMP_index):Spike(), OpenMP_index(NewOpenMP_index) {
}
   	
PropagatedSpike::PropagatedSpike(double NewTime, Neuron * NewSource, int NewPropagationDelayIndex, int NewNPropagationDelayIndex, int NewOpenMP_index): Spike(NewTime,NewSource), propagationDelayIndex(NewPropagationDelayIndex), NPropagationDelayIndex(NewNPropagationDelayIndex), OpenMP_index(NewOpenMP_index){
	inter=NewSource->PropagationStructure->OutputConnectionsWithEquealDealy[NewOpenMP_index][NewPropagationDelayIndex];
	NSynapses=NewSource->PropagationStructure->NSynapsesWithEqualDelay[NewOpenMP_index][NewPropagationDelayIndex];
}

PropagatedSpike::PropagatedSpike(double NewTime, Neuron * NewSource, int NewPropagationDelayIndex, int NewNPropagationDelayIndex, int NewOpenMP_index, Interconnection * NewInter): Spike(NewTime,NewSource), propagationDelayIndex(NewPropagationDelayIndex), NPropagationDelayIndex(NewNPropagationDelayIndex), OpenMP_index(NewOpenMP_index), inter(NewInter){
	NSynapses=NewSource->PropagationStructure->NSynapsesWithEqualDelay[NewOpenMP_index][NewPropagationDelayIndex];
}
   		
PropagatedSpike::~PropagatedSpike(){
}

int PropagatedSpike::GetPropagationDelayIndex (){
	return this->propagationDelayIndex;
}
 
int PropagatedSpike::GetNPropagationDelayIndex (){
	return this->NPropagationDelayIndex;
}

   	



//Optimized function. This function propagates all neuron output spikes that have the same delay.
void PropagatedSpike::ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction){

	if(RealTimeRestriction<2){

		CurrentSimulation->IncrementTotalPropagateEventCounter(OpenMP_index); /*asdfgf*/
		CurrentSimulation->IncrementTotalPropagateCounter(OpenMP_index,NSynapses);

		Neuron * TargetNeuron;
		InternalSpike * Generated;
		LearningRule * ConnectionRule;


		for(int i=0; i<NSynapses; i++){
			TargetNeuron = inter->GetTarget();  // target of the spike

			Generated = TargetNeuron->GetNeuronModel()->ProcessInputSpike(inter, TargetNeuron, this->time);

			if (Generated!=0){
				CurrentSimulation->GetQueue()->InsertEvent(OpenMP_index,Generated);
			}

			CurrentSimulation->WriteState(this->time, TargetNeuron);


			ConnectionRule = inter->GetWeightChange_withoutPost();
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
		if(NPropagationDelayIndex>(propagationDelayIndex+1)){
			PropagatedSpike * spike = new PropagatedSpike(this->GetTime() - this->GetSource()->PropagationStructure->SynapseDelay[OpenMP_index][propagationDelayIndex] + this->GetSource()->PropagationStructure->SynapseDelay[OpenMP_index][propagationDelayIndex+1], this->GetSource(), propagationDelayIndex+1, NPropagationDelayIndex, OpenMP_index);
			CurrentSimulation->GetQueue()->InsertEvent(OpenMP_index,spike);
		} 

	}
}




//Optimized function. This function propagates all neuron output spikes that have the same delay.
void PropagatedSpike::ProcessEvent(Simulation * CurrentSimulation){

	CurrentSimulation->IncrementTotalPropagateEventCounter(OpenMP_index); /*asdfgf*/
	CurrentSimulation->IncrementTotalPropagateCounter(OpenMP_index,NSynapses);

	for(int i=0; i<NSynapses; i++){
		Neuron * TargetNeuron = inter->GetTarget();  // target of the spike

		InternalSpike * Generated = TargetNeuron->GetNeuronModel()->ProcessInputSpike(inter, TargetNeuron, this->time);

		if (Generated!=0){
			CurrentSimulation->GetQueue()->InsertEvent(OpenMP_index,Generated);
		}

		CurrentSimulation->WriteState(this->time, TargetNeuron);

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
	if(NPropagationDelayIndex>(propagationDelayIndex+1)){
		PropagatedSpike * spike = new PropagatedSpike(this->GetTime() - this->GetSource()->PropagationStructure->SynapseDelay[OpenMP_index][propagationDelayIndex] + this->GetSource()->PropagationStructure->SynapseDelay[OpenMP_index][propagationDelayIndex+1], this->GetSource(), propagationDelayIndex+1, NPropagationDelayIndex, OpenMP_index);
		CurrentSimulation->GetQueue()->InsertEvent(OpenMP_index,spike);
	} 
}


int PropagatedSpike::GetOpenMP_index() const{
	return OpenMP_index;
}

void PropagatedSpike::PrintType(){
	cout<<"PropagatedSpike"<<endl;
}

