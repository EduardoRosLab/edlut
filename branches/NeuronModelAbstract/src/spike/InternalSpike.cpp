/***************************************************************************
 *                           InternalSpike.cpp                             *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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

#include "../../include/spike/InternalSpike.h"

#include "../../include/spike/Neuron.h"

#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/communication/OutputSpikeDriver.h"

InternalSpike::InternalSpike():Spike() {
}
   	
InternalSpike::InternalSpike(double NewTime, Neuron * NewSource): Spike(NewTime,NewSource){
}
   		
InternalSpike::~InternalSpike(){
}

void InternalSpike::ProcessEvent(Simulation * CurrentSimulation){
	
	Neuron * neuron=this->source;  // source of the spike
	
	if(!neuron->GetNeuronModel()->DiscardSpike(this)){
		// If it is a valid spike (not discard), generate the next spike in this cell.
		neuron->GetNeuronModel()->GenerateNextSpike(this);
		
		CurrentSimulation->WriteSpike(this);
		CurrentSimulation->WriteState(neuron->GetNeuronState()->GetLastUpdateTime(), this->GetSource());

		// Generate the output activity
		if (neuron->IsOutputConnected()){
			PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + neuron->GetOutputConnectionAt(0)->GetDelay(), neuron, 0);
			CurrentSimulation->GetQueue()->InsertEvent(spike);
		}
    }
}

   	
