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

TimeEventOneNeuron::TimeEventOneNeuron(double NewTime, TimeDrivenNeuronModel * newNeuronModel, Neuron ** newNeurons, int indexNeuron) : Event(NewTime), neuronModel(newNeuronModel), neurons(newNeurons), IndexNeuron(indexNeuron) {

}

TimeEventOneNeuron::~TimeEventOneNeuron(){

}

//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventOneNeuron::ProcessEvent(Simulation * CurrentSimulation, volatile int * RealTimeRestriction){

	//TimeDrivenNeuronModel * neuronModel = (TimeDrivenNeuronModel *) CurrentNetwork->GetNeuronModelAt(this->GetIndexNeuronModel());
	VectorNeuronState * State=neuronModel->GetVectorNeuronState();
	////Updating all cell when using IndexNeuron=-1.
	neuronModel->UpdateState(GetIndexNeuron(), State, this->GetTime());
	//	
	Neuron * Cell = neurons[GetIndexNeuron()];
	if(*RealTimeRestriction<3){

		//CPU write in this array if an internal spike must be generated.
		bool * generateInternalSpike=State->getInternalSpike();

		if(generateInternalSpike[GetIndexNeuron()]==true){
			InternalSpike * internalSpike=new InternalSpike(this->GetTime(),Cell);
			internalSpike->ProcessEvent(CurrentSimulation, RealTimeRestriction);
			delete internalSpike;
		}
		if (Cell->IsMonitored()){
			CurrentSimulation->WriteState(this->GetTime(), Cell);
		}

		//Next TimeEvent for this cell
		CurrentSimulation->GetQueue()->InsertEvent(Cell->get_OpenMP_queue_index(), new TimeEventOneNeuron(this->GetTime() + neuronModel->integrationMethod->PredictedElapsedTime[IndexNeuron], GetModel(), GetNeurons(), GetIndexNeuron()));
	}else{
		//Next TimeEvent for this cell
		CurrentSimulation->GetQueue()->InsertEvent(Cell->get_OpenMP_queue_index(), new TimeEventOneNeuron(this->GetTime() + neuronModel->integrationMethod->PredictedElapsedTime[IndexNeuron], GetModel(), GetNeurons(), GetIndexNeuron()));
	}
}

//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventOneNeuron::ProcessEvent(Simulation * CurrentSimulation){

	//TimeDrivenNeuronModel * neuronModel = (TimeDrivenNeuronModel *) CurrentNetwork->GetNeuronModelAt(this->GetIndexNeuronModel());
	VectorNeuronState * State=neuronModel->GetVectorNeuronState();
	////Updating all cell when using IndexNeuron=-1.
	neuronModel->UpdateState(GetIndexNeuron(), State, this->GetTime());
	//	
	Neuron * Cell = neurons[GetIndexNeuron()];

	//CPU write in this array if an internal spike must be generated.
	bool * generateInternalSpike=State->getInternalSpike();

	if(generateInternalSpike[GetIndexNeuron()]==true){
		InternalSpike * internalSpike=new InternalSpike(this->GetTime(),Cell);
		internalSpike->ProcessEvent(CurrentSimulation);
		delete internalSpike;
	}
	if (Cell->IsMonitored()){
		CurrentSimulation->WriteState(this->GetTime(), Cell);
	}

	//Next TimeEvent for this cell
	CurrentSimulation->GetQueue()->InsertEvent(Cell->get_OpenMP_queue_index(), new TimeEventOneNeuron(this->GetTime() + neuronModel->integrationMethod->PredictedElapsedTime[IndexNeuron], GetModel(), GetNeurons(), GetIndexNeuron()));
}

TimeDrivenNeuronModel * TimeEventOneNeuron::GetModel(){
	return neuronModel;
}

Neuron ** TimeEventOneNeuron::GetNeurons(){
	return neurons;
}

int TimeEventOneNeuron::GetIndexNeuron(){
	return IndexNeuron;
}


void TimeEventOneNeuron::PrintType(){
	cout<<"TimeEventOneNeuron"<<endl;
}

int TimeEventOneNeuron::ProcessingPriority(){
	return 7;
}