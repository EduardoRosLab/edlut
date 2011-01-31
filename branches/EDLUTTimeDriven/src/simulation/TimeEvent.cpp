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

	// Process the neuron state of all the time-driven cells.
	for (int i=0; i<CurrentSimulation->GetNetwork()->GetTimeDrivenNeuronNumber(); ++i){
		Neuron * Cell = CurrentSimulation->GetNetwork()->GetTimeDrivenNeuronAt(i);
		TimeDrivenNeuronModel * NeuronModel = (TimeDrivenNeuronModel *) Cell->GetNeuronModel();
		if(NeuronModel->UpdateState(Cell->GetNeuronState(),this->GetTime())){
			CurrentSimulation->GetQueue()->InsertEvent(new InternalSpike(this->GetTime(),CurrentSimulation->GetNetwork()->GetTimeDrivenNeuronAt(i)));
		}
		CurrentSimulation->WriteState(this->GetTime(), Cell);
	}

	if (CurrentSimulation->GetTimeDrivenStep()>0){
		CurrentSimulation->GetQueue()->InsertEvent(new TimeEvent(this->GetTime()+CurrentSimulation->GetTimeDrivenStep()));
	}
}
