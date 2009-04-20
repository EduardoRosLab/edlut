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
#include "../../include/simulation/Simulation.h"

#include "../../include/communication/OutputSpikeDriver.h"

InputSpike::InputSpike():Spike() {
}
   	
InputSpike::InputSpike(double NewTime, Neuron * NewSource): Spike(NewTime,NewSource){
}
   		
InputSpike::~InputSpike(){
}

void InputSpike::ProcessEvent(Simulation * CurrentSimulation){
	
	Neuron * neuron=this->source;  // source of the spike
    
    CurrentSimulation->WriteSpike(this);
	
	CurrentSimulation->WritePotential(neuron->GetLastUpdate(), this->GetSource(), neuron->GetStateVarAt(1));
		
    //spike.time+=Net.inters[neuron->outconind].delay;
	neuron->GenerateOutputActivity((Spike *) this,CurrentSimulation->GetQueue());
}

   	
