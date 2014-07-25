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
   	
PropagatedSpike::PropagatedSpike(double NewTime, Neuron * NewSource, int NewTarget, int NewOpenMP_index): Spike(NewTime,NewSource), target(NewTarget), OpenMP_index(NewOpenMP_index){
	inter=NewSource->GetOutputConnectionAt(NewOpenMP_index,NewTarget);
}

PropagatedSpike::PropagatedSpike(double NewTime, Neuron * NewSource, int NewTarget, int NewOpenMP_index, Interconnection * NewInter): Spike(NewTime,NewSource), target(NewTarget), OpenMP_index(NewOpenMP_index), inter(NewInter){
}
   		
PropagatedSpike::~PropagatedSpike(){
}

int PropagatedSpike::GetTarget () const{
	return this->target;
}
   		
void PropagatedSpike::SetTarget (int NewTarget) {
	this->target = NewTarget;
}

void PropagatedSpike::IncrementTarget() {
	this->target++;
}

   	

////Optimized function. This function propagates all neuron output spikes that have the same delay.
//void PropagatedSpike::ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction){
//
//	if(!RealTimeRestriction){
//		double CurrentTime = this->GetTime();
//
//		Neuron * source = this->source;
//		Neuron * target;
//		InternalSpike * Generated;
//		LearningRule * ConnectionRule;
//
//		int TargetNum = this->GetTarget();
//
//		Interconnection * inter = this->source->GetOutputConnectionAt(this->GetOpenMP_index(), TargetNum);
//
//
//		//Reference delay
//		double delay=inter->GetDelay();
//
//		while(1){	
//			target = inter->GetTarget();  // target of the spike
//
//			Generated = target->GetNeuronModel()->ProcessInputSpike(inter, target, CurrentTime);
//
//			if (Generated!=0){
//				CurrentSimulation->GetQueue()->InsertEvent(Generated->GetSource()->get_OpenMP_index(),Generated);
//			}
//
//			CurrentSimulation->WriteState(CurrentTime, target);
//
//
//			ConnectionRule = inter->GetWeightChange_withoutPost();
//			// If learning, change weights
//			if(ConnectionRule != 0){
//				ConnectionRule->ApplyPreSynapticSpike(inter,CurrentTime);
//
//			}
//			ConnectionRule = inter->GetWeightChange_withPost();
//			// If learning, change weights
//			if(ConnectionRule != 0){
//				ConnectionRule->ApplyPreSynapticSpike(inter,CurrentTime);
//			}
//
//			// If there are more output connection
//			if(source->GetOutputNumber(this->GetOpenMP_index()) > TargetNum+1){
//				inter = source->GetOutputConnectionAt(this->GetOpenMP_index(),TargetNum+1);
//				// If delays are different
//				if(inter->GetDelay()!=delay){
//					double NextSpikeTime = CurrentTime - delay + inter->GetDelay();
//					PropagatedSpike * nextspike = new PropagatedSpike(NextSpikeTime,source,TargetNum+1,this->GetOpenMP_index());
//					CurrentSimulation->GetQueue()->InsertEvent(this->GetOpenMP_index(),nextspike);
//					break;
//				}
//			}else{
//				break;
//			}
//			// Selecting next output connection
//			SetTarget(TargetNum+1);
//			TargetNum++;
//		}
//
//	}
//}

////Optimized function. This function propagates all neuron output spikes that have the same delay.
//void PropagatedSpike::ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction){
//
//	if(!RealTimeRestriction){
//		double CurrentTime = this->GetTime();
//		int OutputNumber=source->GetOutputNumber(this->GetOpenMP_index());
//
//		Neuron * target;
//		InternalSpike * Generated;
//		LearningRule * ConnectionRule;
//
//
//		int TargetNum = this->GetTarget();
//
//		Interconnection ** ListInter = this->source->GetOutputConnectionAt(this->GetOpenMP_index());
//		Interconnection * inter=ListInter[TargetNum];
//
//
//		//Reference delay
//		double delay=inter->GetDelay();
//
//		while(1){	
//			target = inter->GetTarget();  // target of the spike
//
//			Generated = target->GetNeuronModel()->ProcessInputSpike(inter, target, CurrentTime);
//
//			if (Generated!=0){
//				CurrentSimulation->GetQueue()->InsertEvent(Generated->GetSource()->get_OpenMP_index(),Generated);
//			}
//
//			CurrentSimulation->WriteState(CurrentTime, target);
//
//
//			ConnectionRule = inter->GetWeightChange_withoutPost();
//			// If learning, change weights
//			if(ConnectionRule != 0){
//				ConnectionRule->ApplyPreSynapticSpike(inter,CurrentTime);
//
//			}
//			ConnectionRule = inter->GetWeightChange_withPost();
//			// If learning, change weights
//			if(ConnectionRule != 0){
//				ConnectionRule->ApplyPreSynapticSpike(inter,CurrentTime);
//			}
//
//			// If there are more output connection
//			if( OutputNumber > TargetNum+1){
//				inter = ListInter[TargetNum+1];
//				// If delays are different
//				if(inter->GetDelay()!=delay){
//					double NextSpikeTime = CurrentTime - delay + inter->GetDelay();
//					PropagatedSpike * nextspike = new PropagatedSpike(NextSpikeTime,source,TargetNum+1,this->GetOpenMP_index());
//					CurrentSimulation->GetQueue()->InsertEvent(this->GetOpenMP_index(),nextspike);
//					break;
//				}
//			}else{
//				break;
//			}
//			// Selecting next output connection
//			SetTarget(TargetNum+1);
//			TargetNum++;
//		}
//
//	}
//}

////Optimized function. This function propagates all neuron output spikes that have the same delay.
//void PropagatedSpike::ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction){
//
//	if(!RealTimeRestriction){
//		double CurrentTime = this->GetTime();
//		int OutputNumber=source->GetOutputNumber(this->GetOpenMP_index());
//
//		Neuron * TargetNeuron;
//		InternalSpike * Generated;
//		LearningRule * ConnectionRule;
//
//
//		Interconnection ** ListInter = this->source->GetOutputConnectionAt(this->GetOpenMP_index());
//		Interconnection * inter=ListInter[this->target];
//
//
//		//Reference delay
//		double delay=inter->GetDelay();
//
//		while(1){	
//			TargetNeuron = inter->GetTarget();  // target of the spike
//
//			Generated = TargetNeuron->GetNeuronModel()->ProcessInputSpike(inter, TargetNeuron, CurrentTime);
//
//			if (Generated!=0){
//				CurrentSimulation->GetQueue()->InsertEvent(Generated->GetSource()->get_OpenMP_index(),Generated);
//			}
//
//			CurrentSimulation->WriteState(CurrentTime, TargetNeuron);
//
//
//			ConnectionRule = inter->GetWeightChange_withoutPost();
//			// If learning, change weights
//			if(ConnectionRule != 0){
//				ConnectionRule->ApplyPreSynapticSpike(inter,CurrentTime);
//
//			}
//			ConnectionRule = inter->GetWeightChange_withPost();
//			// If learning, change weights
//			if(ConnectionRule != 0){
//				ConnectionRule->ApplyPreSynapticSpike(inter,CurrentTime);
//			}
//
//			// If there are more output connection
//			if( OutputNumber > target+1){
//				inter = ListInter[target+1];
//				// If delays are different
//				if(inter->GetDelay()!=delay){
//					double NextSpikeTime = CurrentTime - delay + inter->GetDelay();
//					PropagatedSpike * nextspike = new PropagatedSpike(NextSpikeTime,source,target+1,this->GetOpenMP_index());
//					CurrentSimulation->GetQueue()->InsertEvent(this->GetOpenMP_index(),nextspike);
//					break;
//				}
//			}else{
//				break;
//			}
//			// Selecting next output connection
//			IncrementTarget();
//		}
//
//	}
//}

//Optimized function. This function propagates all neuron output spikes that have the same delay.
void PropagatedSpike::ProcessEvent(Simulation * CurrentSimulation, volatile int * RealTimeRestriction){

	if(*RealTimeRestriction<2){

		CurrentSimulation->IncrementTotalPropagateCounter(omp_get_thread_num()); 
		
		int OutputNumber=source->GetOutputNumber(this->GetOpenMP_index());

		Neuron * TargetNeuron;
		InternalSpike * Generated;
		LearningRule * ConnectionRule;


		//Reference delay
		double delay=inter->GetDelay();


		while(1){
			TargetNeuron = inter->GetTarget();  // target of the spike

			Generated = TargetNeuron->GetNeuronModel()->ProcessInputSpike(inter, TargetNeuron, this->time);

			if (Generated!=0){
				CurrentSimulation->GetQueue()->InsertEvent(TargetNeuron->get_OpenMP_queue_index(),Generated);
			}

			CurrentSimulation->WriteState(this->time, TargetNeuron);

			if(*RealTimeRestriction<1){
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
			}
			// If there are more output connection
			if( OutputNumber > target+1){
				inter++;
				// If delays are different
				if(inter->GetDelay()!=delay){
					double NextSpikeTime = this->time - delay + inter->GetDelay();
					PropagatedSpike * nextspike = new PropagatedSpike(NextSpikeTime,source,target+1,this->GetOpenMP_index(),inter);
					CurrentSimulation->GetQueue()->InsertEvent(this->GetOpenMP_index(),nextspike);
					break;
				}
			}else{
				break;
			}
			// Selecting next output connection
			IncrementTarget();
		}

	}
}


//Optimized function. This function propagates all neuron output spikes that have the same delay.
void PropagatedSpike::ProcessEvent(Simulation * CurrentSimulation){


	CurrentSimulation->IncrementTotalPropagateCounter(omp_get_thread_num()); /*asdfgf*/
	
	int OutputNumber=source->GetOutputNumber(this->GetOpenMP_index());

	Neuron * TargetNeuron;
	InternalSpike * Generated;
	LearningRule * ConnectionRule;


	//Reference delay
	double delay=inter->GetDelay();


	while(1){
		TargetNeuron = inter->GetTarget();  // target of the spike

		Generated = TargetNeuron->GetNeuronModel()->ProcessInputSpike(inter, TargetNeuron, this->time);

		if (Generated!=0){
			CurrentSimulation->GetQueue()->InsertEvent(TargetNeuron->get_OpenMP_queue_index(),Generated);
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

		// If there are more output connection
		if( OutputNumber > target+1){
			inter++;
			// If delays are different
			if(inter->GetDelay()!=delay){
				double NextSpikeTime = this->time - delay + inter->GetDelay();
				PropagatedSpike * nextspike = new PropagatedSpike(NextSpikeTime,source,target+1,this->GetOpenMP_index(),inter);
				CurrentSimulation->GetQueue()->InsertEvent(this->GetOpenMP_index(),nextspike);
				break;
			}
		}else{
			break;
		}
		// Selecting next output connection
		IncrementTarget();
	}
}




int PropagatedSpike::GetOpenMP_index() const{
	return OpenMP_index;
}

void PropagatedSpike::PrintType(){
	cout<<"PropagatedSpike"<<endl;
}

