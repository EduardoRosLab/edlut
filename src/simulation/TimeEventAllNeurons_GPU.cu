/***************************************************************************
 *                           TimeEventAllNeurons_GPU.cpp                   *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#include "../../include/simulation/TimeEventAllNeurons_GPU.h"
#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/VectorNeuronState_GPU.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU.h"
		
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"

		

TimeEventAllNeurons_GPU::TimeEventAllNeurons_GPU(double NewTime, int indexNeuronModel) : Event(NewTime), IndexNeuronModel(indexNeuronModel) {

}

TimeEventAllNeurons_GPU::~TimeEventAllNeurons_GPU(){

}

//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventAllNeurons_GPU::ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction){

	Network * CurrentNetwork = CurrentSimulation->GetNetwork();

	double CurrentTime = this->GetTime();

	int * N_TimeDrivenNeuronGPU=CurrentNetwork->GetTimeDrivenNeuronNumberGPU();
	
	InternalSpike * internalSpike;

	TimeDrivenNeuronModel_GPU * neuronModel = (TimeDrivenNeuronModel_GPU *) CurrentNetwork->GetNeuronModelAt(this->GetIndexNeuronModel());
	VectorNeuronState * State=neuronModel->GetVectorNeuronState();
	//Updating all cell when using index=-1.
	neuronModel->UpdateState(-1,State, CurrentTime);

	if(!RealTimeRestriction){

		VectorNeuronState_GPU * state=(VectorNeuronState_GPU *) State;
		//GPU write in this array if an internal spike must be generated.
		bool * generateInternalSpike=state->getInternalSpike();
		Neuron * Cell;

		//We check if some neuron inside the model is monitored
		if(neuronModel->GetVectorNeuronState()->Get_Is_Monitored()){
			for (int t=0; t<N_TimeDrivenNeuronGPU[this->GetIndexNeuronModel()]; t++){
				Cell = CurrentNetwork->GetTimeDrivenNeuronGPUAt(this->GetIndexNeuronModel(),t);
				if(generateInternalSpike[t]==true){
					internalSpike=new InternalSpike(CurrentTime,Cell);
					internalSpike->ProcessEvent(CurrentSimulation, false);
					delete internalSpike;
				}
				if (Cell->IsMonitored()){
					CurrentSimulation->WriteState(CurrentTime, Cell);
				}
			}
		}else{
			for (int t=0; t<N_TimeDrivenNeuronGPU[this->GetIndexNeuronModel()]; t++){
				if(generateInternalSpike[t]==true){
					Cell = CurrentNetwork->GetTimeDrivenNeuronGPUAt(this->GetIndexNeuronModel(),t);
					internalSpike=new InternalSpike(CurrentTime,Cell);
					internalSpike->ProcessEvent(CurrentSimulation, false);
					delete internalSpike;
				}
			}
		}

		CurrentSimulation->GetQueue()->InsertEvent(new TimeEventAllNeurons_GPU(CurrentTime + neuronModel->GetTimeDrivenStep_GPU(), this->GetIndexNeuronModel()));

	}else{
		CurrentSimulation->GetQueue()->InsertEvent(new TimeEventAllNeurons_GPU(CurrentTime + neuronModel->GetTimeDrivenStep_GPU(), this->GetIndexNeuronModel()));
	}
}

int TimeEventAllNeurons_GPU::GetIndexNeuronModel(){
	return IndexNeuronModel;
}