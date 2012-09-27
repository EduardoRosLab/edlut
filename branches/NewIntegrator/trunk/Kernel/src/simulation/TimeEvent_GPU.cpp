/***************************************************************************
 *                           TimeEvent_GPU.cpp                              *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/simulation/TimeEvent_GPU.h"
#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/VectorNeuronState_GPU.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"
		
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"

		

TimeEvent_GPU::TimeEvent_GPU(double NewTime) : Event(NewTime) {

}

TimeEvent_GPU::~TimeEvent_GPU(){

}

//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEvent_GPU::ProcessEvent(Simulation * CurrentSimulation){

	Network * CurrentNetwork = CurrentSimulation->GetNetwork();

	double CurrentTime = this->GetTime();

	int nnutypes=CurrentNetwork->GetNneutypes();
	int * N_TimeDrivenNeuronGPU=CurrentNetwork->GetTimeDrivenNeuronNumberGPU();
	
	InternalSpike * internalSpike;

	for(int z=0; z<nnutypes; z++){
		if(N_TimeDrivenNeuronGPU[z]>0){
			TimeDrivenNeuronModel * neuronModel = (TimeDrivenNeuronModel *) CurrentNetwork->GetNeuronModelAt(z);
			VectorNeuronState * State=neuronModel->GetVectorNeuronState();
			//Updating all cell when using index=-1.
			neuronModel->UpdateState(-1,State, CurrentTime);

			VectorNeuronState_GPU * state=(VectorNeuronState_GPU *) State;
			//GPU write in this array if an internal spike must be generated.
			bool * generateInternalSpike=state->getInternalSpike();
			Neuron * Cell;

			//We check if some neuron inside the model is monitored
			if(neuronModel->GetVectorNeuronState()->Get_Is_Monitored()){
				for (int t=0; t<N_TimeDrivenNeuronGPU[z]; t++){
					Cell = CurrentNetwork->GetTimeDrivenNeuronGPUAt(z,t);
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
				for (int t=0; t<N_TimeDrivenNeuronGPU[z]; t++){
					if(generateInternalSpike[t]==true){
						Cell = CurrentNetwork->GetTimeDrivenNeuronGPUAt(z,t);
						internalSpike=new InternalSpike(CurrentTime,Cell);
						internalSpike->ProcessEvent(CurrentSimulation);
						delete internalSpike;
					}
				}
			}
		}
	}


	float TimeDrivenStepGPU = CurrentSimulation->GetTimeDrivenStepGPU();

	if (TimeDrivenStepGPU>0){
		CurrentSimulation->GetQueue()->InsertEvent(new TimeEvent_GPU(CurrentTime+TimeDrivenStepGPU));
	}
}