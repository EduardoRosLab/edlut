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

#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"

TimeEvent::TimeEvent(double NewTime) : Event(NewTime) {

}

TimeEvent::~TimeEvent(){

}

void TimeEvent::ProcessEvent(Simulation * CurrentSimulation){

	Network * CurrentNetwork = CurrentSimulation->GetNetwork();

	long int TDCells = CurrentNetwork->GetTimeDrivenNeuronNumber();

	float CurrentTime = this->GetTime();

	// Process the neuron state of all the time-driven cells.
	for (int i=0; i<TDCells; ++i){
		Neuron * Cell = CurrentNetwork->GetTimeDrivenNeuronAt(i);
		TimeDrivenNeuronModel * NeuronModel = (TimeDrivenNeuronModel *) Cell->GetNeuronModel();
		if(NeuronModel->UpdateState(Cell->GetNeuronState(),CurrentTime)){
			CurrentSimulation->GetQueue()->InsertEvent(new InternalSpike(CurrentTime,Cell));
		}

		if (Cell->IsMonitored()){
			CurrentSimulation->WriteState(CurrentTime, Cell);
		}
	}

	float TimeDrivenStep = CurrentSimulation->GetTimeDrivenStep();

	if (TimeDrivenStep>0){
		CurrentSimulation->GetQueue()->InsertEvent(new TimeEvent(CurrentTime+TimeDrivenStep));
	}
}
