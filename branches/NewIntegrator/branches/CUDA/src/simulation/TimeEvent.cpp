/***************************************************************************
 *                           TimeEvent.cpp                                 *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
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

#include "../../include/simulation/TimeEvent.h"
#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"

TimeEvent::TimeEvent(double NewTime) : Event(NewTime) {

}

TimeEvent::~TimeEvent(){

}



//void TimeEvent::ProcessEvent(Simulation * CurrentSimulation){
//
//	Network * CurrentNetwork = CurrentSimulation->GetNetwork();
//
//	double CurrentTime = this->GetTime();
//
//	int nnutypes=CurrentNetwork->GetNneutypes();
//	int * N_TimeDrivenNeuron=CurrentNetwork->GetTimeDrivenNeuronNumber();
//	
//	InternalSpike * internalSpike;
//
//	for(int z=0; z<nnutypes; z++){
//		if(N_TimeDrivenNeuron[z]>0){
//			TimeDrivenNeuronModel * neuronModel = (TimeDrivenNeuronModel *) CurrentNetwork->GetNeuronModelAt(z);
//			VectorNeuronState * State=neuronModel->GetVectorNeuronState();
//			//Updating all cell when using index=-1.
//			neuronModel->UpdateState(-1,State, CurrentTime);
//
//			//CPU write in this array if an internal spike must be generated.
//			bool * generateInternalSpike=State->getInternalSpike();
//			Neuron * Cell;
//
//			if(neuronModel->GetVectorNeuronState()->Get_Is_Monitored()){
//				for (int t=0; t<N_TimeDrivenNeuron[z]; t++){
//					Cell = CurrentNetwork->GetTimeDrivenNeuronAt(z,t);
//					if(generateInternalSpike[t]==true){
//						CurrentSimulation->GetQueue()->InsertEvent(new InternalSpike(CurrentTime,Cell));
//					}
//					if (Cell->IsMonitored()){
//						CurrentSimulation->WriteState(CurrentTime, Cell);
//					}
//				}
//			}else{
//				for (int t=0; t<N_TimeDrivenNeuron[z]; t++){
//					if(generateInternalSpike[t]==true){
//						Cell = CurrentNetwork->GetTimeDrivenNeuronAt(z,t);
//						CurrentSimulation->GetQueue()->InsertEvent(new InternalSpike(CurrentTime,Cell));
//					}
//				}
//			}
//		}
//	}
//
//
//	float TimeDrivenStep = CurrentSimulation->GetTimeDrivenStep();
//
//	if (TimeDrivenStep>0){
//		CurrentSimulation->GetQueue()->InsertEvent(new TimeEvent(CurrentTime+TimeDrivenStep));
//	}
//}


//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEvent::ProcessEvent(Simulation * CurrentSimulation){

	Network * CurrentNetwork = CurrentSimulation->GetNetwork();

	double CurrentTime = this->GetTime();

	int nnutypes=CurrentNetwork->GetNneutypes();
	int * N_TimeDrivenNeuron=CurrentNetwork->GetTimeDrivenNeuronNumber();
	
	InternalSpike * internalSpike;

	for(int z=0; z<nnutypes; z++){
		if(N_TimeDrivenNeuron[z]>0){
			TimeDrivenNeuronModel * neuronModel = (TimeDrivenNeuronModel *) CurrentNetwork->GetNeuronModelAt(z);
			VectorNeuronState * State=neuronModel->GetVectorNeuronState();
			//Updating all cell when using index=-1.
			neuronModel->UpdateState(-1,State, CurrentTime);

			Neuron * Cell;

			if(neuronModel->GetVectorNeuronState()->Get_Is_Monitored()){
				//CPU write in this array if an internal spike must be generated.
				bool * generateInternalSpike=State->getInternalSpike();
				
				for (int t=0; t<N_TimeDrivenNeuron[z]; t++){
					Cell = CurrentNetwork->GetTimeDrivenNeuronAt(z,t);
					if(generateInternalSpike[t]==true){
						internalSpike=new InternalSpike(CurrentTime,Cell);
						internalSpike->ProcessEvent(CurrentSimulation);
						delete internalSpike;
					}
					if (Cell->IsMonitored()){
						CurrentSimulation->WriteState(CurrentTime, Cell);
					}
				}
			}else{
				//CPU write in this array if an internal spike must be generated.
				bool * generateInternalSpike=State->getInternalSpike();

				for (int t=0; t<N_TimeDrivenNeuron[z]; t++){
					if(generateInternalSpike[t]==true){
						Cell = CurrentNetwork->GetTimeDrivenNeuronAt(z,t);
						internalSpike=new InternalSpike(CurrentTime,Cell);
						internalSpike->ProcessEvent(CurrentSimulation);
						delete internalSpike;
					}
				}
			}
		}
	}


	float TimeDrivenStep = CurrentSimulation->GetTimeDrivenStep();

	if (TimeDrivenStep>0){
		CurrentSimulation->GetQueue()->InsertEvent(new TimeEvent(CurrentTime+TimeDrivenStep));
	}
}