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

   	
TableBasedModelHFEvent::TableBasedModelHFEvent(double NewTime, int NewMaxSize): InternalSpike(NewTime, NULL), MaxSize(NewMaxSize), NElements(0){
	Neurons= (Neuron**) new Neuron * [MaxSize];
}
   		
TableBasedModelHFEvent::~TableBasedModelHFEvent(){
}

void TableBasedModelHFEvent::ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction){

	if(RealTimeRestriction<3){
		TableBasedModelHF * Model = (TableBasedModelHF *) Neurons[0]->GetNeuronModel();

		for(int i=0; i<NElements; i++){
			this->SetSource(Neurons[i]);

			InternalSpike * Generated = Model->ProcessActivityAndPredictSpike(Neurons[i]);
			if(Generated!=0){
				CurrentSimulation->GetQueue()->InsertEvent(Neurons[i]->get_OpenMP_queue_index(),Generated);
			}
		}
	}
}

void TableBasedModelHFEvent::ProcessEvent(Simulation * CurrentSimulation){
	TableBasedModelHF * Model = (TableBasedModelHF *) Neurons[0]->GetNeuronModel();

	for(int i=0; i<NElements; i++){
		this->SetSource(Neurons[i]);

		InternalSpike * Generated = Model->ProcessActivityAndPredictSpike(Neurons[i]);
		if(Generated!=0){
			CurrentSimulation->GetQueue()->InsertEvent(Neurons[i]->get_OpenMP_queue_index(),Generated);
		}
	}
}

void TableBasedModelHFEvent::IncludeNewNeuron(Neuron * neuron){
	Neurons[NElements]=neuron;
	NElements++;
}

   	
void TableBasedModelHFEvent::PrintType(){
	cout<<"TableBasedModelHFEvent"<<endl;
}


enum EventPriority TableBasedModelHFEvent::ProcessingPriority(){
	return TABLEBASEDMODELHFEVENT;
}