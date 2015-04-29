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
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/spike/TimeDrivenInternalSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/NeuronModelPropagationDelayStructure.h"

#include "../../include/simulation/SynchronizeSimulationEvent.h"

#include "../../include/openmp/openmp.h"

		

TimeEventAllNeurons_GPU::TimeEventAllNeurons_GPU(double NewTime, TimeDrivenNeuronModel_GPU * newNeuronModel, Neuron ** newNeurons) : Event(NewTime), neuronModel(newNeuronModel), neurons(newNeurons){
}

TimeEventAllNeurons_GPU::~TimeEventAllNeurons_GPU(){

}


//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventAllNeurons_GPU::ProcessEvent(Simulation * CurrentSimulation , int RealTimeRestriction){

	double CurrentTime = this->GetTime();

	if(RealTimeRestriction<3){
		VectorNeuronState * State=neuronModel->GetVectorNeuronState();

		neuronModel->UpdateState(-1, State, CurrentTime);

		TimeDrivenInternalSpike NewEvent(CurrentTime, State, neuronModel->PropagationStructure, neurons, omp_get_thread_num());
		NewEvent.ProcessEvent(CurrentSimulation, RealTimeRestriction);
	}

	//Next TimeEvent for all cell
	CurrentSimulation->GetQueue()->InsertEvent(new TimeEventAllNeurons_GPU(CurrentTime + neuronModel->TimeDrivenStep_GPU, GetModel(), GetNeurons()));
}

//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventAllNeurons_GPU::ProcessEvent(Simulation * CurrentSimulation){

	double CurrentTime = this->GetTime();

	VectorNeuronState * State=neuronModel->GetVectorNeuronState();

	neuronModel->UpdateState(-1, State, CurrentTime);

	TimeDrivenInternalSpike NewEvent(CurrentTime, State, neuronModel->PropagationStructure, neurons, omp_get_thread_num());
	NewEvent.ProcessEvent(CurrentSimulation);

	//Next TimeEvent for all cell
	CurrentSimulation->GetQueue()->InsertEvent(new TimeEventAllNeurons_GPU(CurrentTime + neuronModel->TimeDrivenStep_GPU, GetModel(), GetNeurons()));
}

TimeDrivenNeuronModel_GPU * TimeEventAllNeurons_GPU::GetModel(){
	return neuronModel;
}

Neuron ** TimeEventAllNeurons_GPU::GetNeurons(){
	return neurons;
}

void TimeEventAllNeurons_GPU::PrintType(){
	cout<<"TimeEventAllNeurons_GPU"<<endl;
}

enum EventPriority TimeEventAllNeurons_GPU::ProcessingPriority(){
	return TIMEEVENT;
}