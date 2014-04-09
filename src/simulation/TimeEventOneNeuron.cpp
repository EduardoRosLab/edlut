/***************************************************************************
 *                           TimeEventOneNeuron.cpp                        *
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

#include "../../include/simulation/TimeEventOneNeuron.h"
#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"

TimeEventOneNeuron::TimeEventOneNeuron(double NewTime, int indexNeuronModel, int indexNeuron) : Event(NewTime), IndexNeuronModel(indexNeuronModel), IndexNeuron(indexNeuron) {

}

TimeEventOneNeuron::~TimeEventOneNeuron(){

}

//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventOneNeuron::ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction){
	Network * CurrentNetwork = CurrentSimulation->GetNetwork();

	double CurrentTime = this->GetTime();

	int * N_TimeDrivenNeuron=CurrentNetwork->GetTimeDrivenNeuronNumber();
	
	InternalSpike * internalSpike;


	TimeDrivenNeuronModel * neuronModel = (TimeDrivenNeuronModel *) CurrentNetwork->GetNeuronModelAt(this->GetIndexNeuronModel());
	VectorNeuronState * State=neuronModel->GetVectorNeuronState();
	//Updating all cell when using IndexNeuron=-1.
	neuronModel->UpdateState(GetIndexNeuron(), State, CurrentTime);

	if(!RealTimeRestriction){

		//CPU write in this array if an internal spike must be generated.
		bool * generateInternalSpike=State->getInternalSpike();

		Neuron * Cell;

		Cell = CurrentNetwork->GetTimeDrivenNeuronAt(GetIndexNeuronModel(),GetIndexNeuron());
		if(generateInternalSpike[GetIndexNeuron()]==true){
			internalSpike=new InternalSpike(CurrentTime,Cell);
			internalSpike->ProcessEvent(CurrentSimulation, false);
			delete internalSpike;
		}
		if (Cell->IsMonitored()){
			CurrentSimulation->WriteState(CurrentTime, Cell);
		}

		//Next TimeEvent for this cell
		CurrentSimulation->GetQueue()->InsertEvent(new TimeEventOneNeuron(CurrentTime + neuronModel->integrationMethod->PredictedElapsedTime[IndexNeuron], GetIndexNeuronModel(), GetIndexNeuron()));
	}else{
		//Next TimeEvent for this cell
		CurrentSimulation->GetQueue()->InsertEvent(new TimeEventOneNeuron(CurrentTime + neuronModel->integrationMethod->PredictedElapsedTime[IndexNeuron], GetIndexNeuronModel(), GetIndexNeuron()));
	}
}

int TimeEventOneNeuron::GetIndexNeuronModel(){
	return IndexNeuronModel;
}

int TimeEventOneNeuron::GetIndexNeuron(){
	return IndexNeuron;
}