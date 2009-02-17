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

#include "./include/InternalSpike.h"

#include <list>
#include "./include/Neuron.h"

#include "../simulation/include/Simulation.h"

#include "../communication/include/OutputSpikeDriver.h"

InternalSpike::InternalSpike():Spike() {
}
   	
InternalSpike::InternalSpike(double NewTime, Neuron * NewSource): Spike(NewTime,NewSource){
}
   		
InternalSpike::~InternalSpike(){
}

void InternalSpike::ProcessEvent(Simulation * CurrentSimulation){
	
	Neuron * neuron=this->source;  // source of the spike
	
	if(neuron->GetPredictedSpike()==this->time){
		
		neuron->ProcessInputActivity(this);
			
		// Not needed for only one spike per input
		neuron->GenerateAutoActivity(CurrentSimulation->GetQueue());
		// Not-needed end (can be replaced by neuron->predictedspike=0)
		
		CurrentSimulation->WriteSpike(this);
		CurrentSimulation->WritePotential(neuron->GetLastUpdate(), this->GetSource(), neuron->GetStateVarAt(1));
		
		//spike.time+=Net.inters[neuron->outconind].delay;
    	neuron->GenerateOutputActivity(this,CurrentSimulation->GetQueue());
    }
}

   	
