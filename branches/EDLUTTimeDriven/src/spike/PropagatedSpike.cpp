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

#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"

#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/learning_rules/LearningRule.h"

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
	Neuron * target = inter->GetTarget();  // target of the spike
	Neuron * source = inter->GetSource();
	
	InternalSpike * Generated = target->GetNeuronModel()->ProcessInputSpike(this);

	if (Generated!=0){
		CurrentSimulation->GetQueue()->InsertEvent(Generated);
	}

	CurrentSimulation->WriteState(this->GetTime(), inter->GetTarget());
	
	// Propagate received spike to the next connection
	if(source->GetOutputNumber() > this->GetTarget()+1){
		Interconnection * NewConnection = source->GetOutputConnectionAt(this->GetTarget()+1);
		PropagatedSpike * nextspike = new PropagatedSpike(this->GetTime()-source->GetOutputConnectionAt(this->GetTarget())->GetDelay()+NewConnection->GetDelay(),source,this->GetTarget()+1);
		CurrentSimulation->GetQueue()->InsertEvent(nextspike);
	}

	inter->SetLastSpikeTime(this->time);

	// If learning, change weights
    if(inter->GetWeightChange() != 0){
    	inter->GetWeightChange()->ApplyPreSynapticSpike(inter,this->time);
    }
}

   	
