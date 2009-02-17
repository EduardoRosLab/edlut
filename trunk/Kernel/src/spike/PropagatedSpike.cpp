/***************************************************************************
 *                           PropagatedSpike.cpp                           *
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

#include "./include/PropagatedSpike.h"

#include "./include/Interconnection.h"
#include "./include/Neuron.h"

#include "../simulation/include/Simulation.h"

PropagatedSpike::PropagatedSpike():Spike() {
}
   	
PropagatedSpike::PropagatedSpike(double NewTime, Neuron * NewSource, int NewTarget): Spike(NewTime,NewSource), target(NewTarget){
}
   		
PropagatedSpike::~PropagatedSpike(){
}

int PropagatedSpike::GetTarget () const{
	return this->target;
}
   		
void PropagatedSpike::SetTarget (int NewTarget){
	this->target = NewTarget;
}

void PropagatedSpike::ProcessEvent(Simulation * CurrentSimulation){
	
	Interconnection * inter = this->source->GetOutputConnectionAt(this->target);
	Neuron * neuron = inter->GetTarget();  // target of the spike
	neuron->ProcessInputSynapticActivity(this);
	
	CurrentSimulation->WritePotential(this->GetTime(), inter->GetTarget(), neuron->GetStateVarAt(1));
	//CurrentSimulation->WriteSpike(this);
	
	neuron->GenerateInputActivity(CurrentSimulation->GetQueue());
	
	this->source->PropagateOutputSpike(this, CurrentSimulation->GetQueue());
	            
    if(inter->GetWeightChange() != 0){
    	inter->ChangeWeights(this->time);
    }
}

   	
