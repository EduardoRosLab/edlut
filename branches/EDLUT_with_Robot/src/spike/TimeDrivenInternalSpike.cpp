/***************************************************************************
 *                           TimeDrivenInternalSpike.cpp                   *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/spike/TimeDrivenInternalSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/spike/Neuron.h"
#include "../../include/spike/PropagatedSpike.h"

#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/EventDrivenNeuronModel.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/communication/OutputSpikeDriver.h"

#include "../../include/learning_rules/LearningRule.h"

#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"

TimeDrivenInternalSpike::TimeDrivenInternalSpike():Spike() {
}
   	
TimeDrivenInternalSpike::TimeDrivenInternalSpike(double NewTime, VectorNeuronState * NewState, Neuron ** NewNeurons) : Spike(NewTime, NULL), State(NewState), Neurons(NewNeurons){
}
   		
TimeDrivenInternalSpike::~TimeDrivenInternalSpike(){
}

void TimeDrivenInternalSpike::ProcessInternalSpikeEvent(Simulation * CurrentSimulation, int index){
	Neuron * neuron = Neurons[index];
	this->SetSource(neuron);
	CurrentSimulation->IncrementTotalSpikeCounter( neuron->get_OpenMP_queue_index()); 

	CurrentSimulation->WriteSpike(this);
	CurrentSimulation->WriteState(neuron->GetVectorNeuronState()->GetLastUpdateTime(neuron->GetIndex_VectorNeuronState()), neuron);


	// Generate the output activity
	for (int i = 0; i<NumberOfOpenMPQueues; i++){
		if (neuron->IsOutputConnected(i)){
			PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + neuron->GetOutputConnectionAt(i, 0)->GetDelay(), neuron, 0, i);
			if (i == neuron->get_OpenMP_queue_index()){
				CurrentSimulation->GetQueue()->InsertEvent(i, spike);
			}
			else{
				CurrentSimulation->GetQueue()->InsertEventInBuffer(neuron->get_OpenMP_queue_index(), i, spike);
			}
		}
	}

	if (neuron->GetInputNumberWithPostSynapticLearning()>0){
		#ifdef _OPENMP 
			#if	_OPENMP >= OPENMPVERSION30
				#pragma omp task
			#endif
		#endif

		{
			for (int i = 0; i < neuron->GetInputNumberWithPostSynapticLearning(); ++i){
				Interconnection * inter = neuron->GetInputConnectionWithPostSynapticLearningAt(i);
				inter->GetWeightChange_withPost()->ApplyPostSynapticSpike(inter, this->time);
			}
		}

	}

}

void TimeDrivenInternalSpike::ProcessEvent(Simulation * CurrentSimulation, volatile int * RealTimeRestriction){

	if(*RealTimeRestriction<2){

		//CPU write in this array if an internal spike must be generated.
		bool * generateInternalSpike = State->getInternalSpike();

		Neuron * Cell;

		//We check if some neuron inside the model is monitored
		if (State->Get_Is_Monitored()){
			for (int t = 0; t<State->GetSizeState(); t++){
				Cell = Neurons[t];
				if (generateInternalSpike[t] == true){
					ProcessInternalSpikeEvent(CurrentSimulation, t);
				}
				if (Cell->IsMonitored()){
					CurrentSimulation->WriteState(this->GetTime(), Cell);
				}
			}
		}
		else{
			for (int t = 0; t<State->GetSizeState(); t++){
				if (generateInternalSpike[t] == true){
					Cell = Neurons[t];
					ProcessInternalSpikeEvent(CurrentSimulation, t);
				}
			}

		}
		#ifdef _OPENMP 
			#if	_OPENMP >= OPENMPVERSION30
				#pragma omp taskwait
			#endif
		#endif
	}
}

void TimeDrivenInternalSpike::ProcessEvent(Simulation * CurrentSimulation){

	//CPU write in this array if an internal spike must be generated.
	bool * generateInternalSpike = State->getInternalSpike();

	Neuron * Cell;

	//We check if some neuron inside the model is monitored
	if (State->Get_Is_Monitored()){
		for (int t = 0; t<State->GetSizeState(); t++){
			Cell = Neurons[t];
			if (generateInternalSpike[t] == true){
				ProcessInternalSpikeEvent(CurrentSimulation, t);
			}
			if (Cell->IsMonitored()){
				CurrentSimulation->WriteState(this->GetTime(), Cell);
			}
		}
	}
	else{
		for (int t = 0; t<State->GetSizeState(); t++){
			if (generateInternalSpike[t] == true){
				Cell = Neurons[t];
				ProcessInternalSpikeEvent(CurrentSimulation, t);
			}
		}

	}
	#ifdef _OPENMP 
		#if	_OPENMP >= OPENMPVERSION30
			#pragma omp taskwait
		#endif
	#endif
}





   	
void TimeDrivenInternalSpike::PrintType(){
	cout<<"TimeDrivenInternalSpike"<<endl;
}
