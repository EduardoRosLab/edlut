/***************************************************************************
 *                           TimeEventAllNeurons.cpp                       *
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

#include "../../include/simulation/TimeEventAllNeurons.h"
#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"



TimeEventAllNeurons::TimeEventAllNeurons(double NewTime, int indexNeuronModel) : TimeEventOneNeuron(NewTime, indexNeuronModel, -1) {

}

TimeEventAllNeurons::~TimeEventAllNeurons(){

}

//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventAllNeurons::ProcessEvent(Simulation * CurrentSimulation){
	Network * CurrentNetwork = CurrentSimulation->GetNetwork();

	double CurrentTime = this->GetTime();

	int * N_TimeDrivenNeuron=CurrentNetwork->GetTimeDrivenNeuronNumber();
	
	InternalSpike * internalSpike;


	TimeDrivenNeuronModel * neuronModel = (TimeDrivenNeuronModel *) CurrentNetwork->GetNeuronModelAt(this->GetIndexNeuronModel());
	VectorNeuronState * State=neuronModel->GetVectorNeuronState();
	//Updating all cell when using IndexNeuron=-1.
	neuronModel->UpdateState(GetIndexNeuron(), State, CurrentTime);

	//CPU write in this array if an internal spike must be generated.
	bool * generateInternalSpike=State->getInternalSpike();

	Neuron * Cell;

	//Updating all cell when using IndexNeuron=-1.
	if(GetIndexNeuron()==-1){
		//We check if some neuron inside the model is monitored
		if(neuronModel->GetVectorNeuronState()->Get_Is_Monitored()){
			for (int t=0; t<N_TimeDrivenNeuron[this->GetIndexNeuronModel()]; t++){
				Cell = CurrentNetwork->GetTimeDrivenNeuronAt(this->GetIndexNeuronModel(),t);
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
			for (int t=0; t<N_TimeDrivenNeuron[this->GetIndexNeuronModel()]; t++){
				if(generateInternalSpike[t]==true){
					Cell = CurrentNetwork->GetTimeDrivenNeuronAt(this->GetIndexNeuronModel(),t);
					internalSpike=new InternalSpike(CurrentTime,Cell);
					internalSpike->ProcessEvent(CurrentSimulation);
					delete internalSpike;
				}
			}
		}

		//Next TimeEvent for all cell
		CurrentSimulation->GetQueue()->InsertEvent(new TimeEventAllNeurons(CurrentTime + neuronModel->integrationMethod->PredictedElapsedTime[0], this->GetIndexNeuronModel()));
	}
}
