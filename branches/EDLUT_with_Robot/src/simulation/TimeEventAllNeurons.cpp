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

#include "../../include/spike/TimeDrivenInternalSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"

TimeEventAllNeurons::TimeEventAllNeurons(double NewTime, TimeDrivenNeuronModel * newNeuronModel, Neuron ** newNeurons) : TimeEventOneNeuron(NewTime, newNeuronModel, newNeurons, -1) {

}

TimeEventAllNeurons::~TimeEventAllNeurons(){

}



//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventAllNeurons::ProcessEvent(Simulation * CurrentSimulation, volatile int * RealTimeRestriction){

	double CurrentTime = this->GetTime();
	
	if(*RealTimeRestriction<3){
		VectorNeuronState * State=neuronModel->GetVectorNeuronState();

		neuronModel->UpdateState(-1, State, CurrentTime);

		TimeDrivenInternalSpike NewEvent(CurrentTime, State, neurons);
		NewEvent.ProcessEvent(CurrentSimulation, RealTimeRestriction);
	}


	//Next TimeEvent for all cell
	CurrentSimulation->GetQueue()->InsertEvent(new TimeEventAllNeurons(CurrentTime + neuronModel->integrationMethod->PredictedElapsedTime[0], GetModel(), GetNeurons()));
}

//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventAllNeurons::ProcessEvent(Simulation * CurrentSimulation){

	double CurrentTime = this->GetTime();

	VectorNeuronState * State=neuronModel->GetVectorNeuronState();

	neuronModel->UpdateState(-1, State, CurrentTime);

	TimeDrivenInternalSpike NewEvent(CurrentTime, State, neurons);
	NewEvent.ProcessEvent(CurrentSimulation);

	//Next TimeEvent for all cell
	CurrentSimulation->GetQueue()->InsertEvent(new TimeEventAllNeurons(CurrentTime + neuronModel->integrationMethod->PredictedElapsedTime[0], GetModel(), GetNeurons()));
}



void TimeEventAllNeurons::PrintType(){
	cout<<"TimeEventAllNeurons"<<endl;
}
