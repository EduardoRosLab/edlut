/***************************************************************************
 *                           TableBasedModelHFEven.cpp                     *
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

#include "../../include/spike/TableBasedModelHFEvent.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/spike/Neuron.h"
#include "../../include/spike/PropagatedSpike.h"

#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/TableBasedModelHF.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/communication/OutputSpikeDriver.h"

#include "../../include/learning_rules/LearningRule.h"

#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"

TableBasedModelHFEvent::TableBasedModelHFEvent():InternalSpike() {
}
   	
TableBasedModelHFEvent::TableBasedModelHFEvent(double NewTime, Neuron * NewSource): InternalSpike(NewTime,NewSource){
}
   		
TableBasedModelHFEvent::~TableBasedModelHFEvent(){
}

void TableBasedModelHFEvent::ProcessEvent(Simulation * CurrentSimulation, volatile int * RealTimeRestriction){

	if(*RealTimeRestriction<3){
		Neuron * neuron=this->GetSource();  // source of the spike

		TableBasedModelHF * Model = (TableBasedModelHF *) neuron->GetNeuronModel();

		InternalSpike * Generated = Model->ProcessActivityAndPredictSpike(neuron);
		if(Generated!=0){
			CurrentSimulation->GetQueue()->InsertEvent(neuron->get_OpenMP_queue_index(),Generated);
		}
	}
}

void TableBasedModelHFEvent::ProcessEvent(Simulation * CurrentSimulation){

	Neuron * neuron=this->GetSource();  // source of the spike

	TableBasedModelHF * Model = (TableBasedModelHF *) neuron->GetNeuronModel();

	InternalSpike * Generated = Model->ProcessActivityAndPredictSpike(neuron);
	if(Generated!=0){
		CurrentSimulation->GetQueue()->InsertEvent(neuron->get_OpenMP_queue_index(),Generated);
	}
}

   	
void TableBasedModelHFEvent::PrintType(){
	cout<<"TableBasedModelHFEvent"<<endl;
}


int TableBasedModelHFEvent::ProcessingPriority(){
	return 8;
}