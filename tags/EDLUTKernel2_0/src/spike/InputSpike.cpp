/***************************************************************************
 *                           InputSpike.cpp                                *
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

#include "../../include/spike/InputSpike.h"

#include "../../include/spike/Neuron.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/Simulation.h"

#include "../../include/neuron_model/NeuronState.h"

#include "../../include/communication/OutputSpikeDriver.h"

InputSpike::InputSpike():Spike() {
}
   	
InputSpike::InputSpike(double NewTime, Neuron * NewSource): Spike(NewTime,NewSource){
}
   		
InputSpike::~InputSpike(){
}

void InputSpike::ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction){

	if(!RealTimeRestriction){
		
		Neuron * neuron=this->source;  // source of the spike
	    
		CurrentSimulation->WriteSpike(this);
		
		// CurrentSimulation->WriteState(neuron->GetVectorNeuronState()->GetLastUpdateTime(), this->GetSource());
			
		if (neuron->IsOutputConnected()){
			PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + neuron->GetOutputConnectionAt(0)->GetDelay(), neuron, 0);
			CurrentSimulation->GetQueue()->InsertEvent(spike);
		}
	}
}

   	
