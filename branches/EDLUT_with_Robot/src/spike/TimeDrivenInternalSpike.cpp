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
#include "../../include/spike/TimeDrivenPropagatedSpike.h"
#include "../../include/spike/NeuronModelPropagationDelayStructure.h"

#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/EventDrivenNeuronModel.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/communication/OutputSpikeDriver.h"

#include "../../include/learning_rules/LearningRule.h"

#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"

TimeDrivenInternalSpike::TimeDrivenInternalSpike(double NewTime, VectorNeuronState * NewState, NeuronModelPropagationDelayStructure * NewPropagationStructure, Neuron ** NewNeurons, int NewOpenMP_index) : Spike(NewTime, NULL), State(NewState), PropagationStructure(NewPropagationStructure), Neurons(NewNeurons), OpenMP_index(NewOpenMP_index){
	timeDrivenPropagatedSpike=(TimeDrivenPropagatedSpike***) new TimeDrivenPropagatedSpike ** [NumberOfOpenMPQueues];
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		timeDrivenPropagatedSpike[i]=(TimeDrivenPropagatedSpike**)new TimeDrivenPropagatedSpike**[PropagationStructure->GetSize(i)];
		for(int j=0; j<PropagationStructure->GetSize(i); j++){
			timeDrivenPropagatedSpike[i][j]=new TimeDrivenPropagatedSpike(NewTime+PropagationStructure->GetDelayAt(i,j),i, PropagationStructure->GetEventSize(i, j));
		}
	}
}
   		
TimeDrivenInternalSpike::~TimeDrivenInternalSpike(){
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		delete timeDrivenPropagatedSpike[i];
	}
	delete timeDrivenPropagatedSpike;
}

void TimeDrivenInternalSpike::ProcessInternalSpikeEvent(Simulation * CurrentSimulation, int index){
	Neuron * neuron = Neurons[index];
	this->SetSource(neuron);
	CurrentSimulation->IncrementTotalSpikeCounter( neuron->get_OpenMP_queue_index()); 

	CurrentSimulation->WriteSpike(this);
	CurrentSimulation->WriteState(neuron->GetVectorNeuronState()->GetLastUpdateTime(neuron->GetIndex_VectorNeuronState()), neuron);


	// Generate the output activity
	for (int i = 0; i<NumberOfOpenMPQueues; i++){
		for (int j=0; j<neuron->PropagationStructure->NDifferentDelays[i]; j++){
			int index=neuron->PropagationStructure->IndexSynapseDelay[i][j];
			//Include the new source and checks if the TimeDrivenPropagatedSpike event is full
			if(timeDrivenPropagatedSpike[i][index]->IncludeNewSource(neuron->PropagationStructure->NSynapsesWithEqualDelay[i][j],neuron->PropagationStructure->OutputConnectionsWithEquealDealy[i][j])){
				if (i == this->OpenMP_index){
					CurrentSimulation->GetQueue()->InsertEvent(i, timeDrivenPropagatedSpike[i][index]);
				}
				else{
					CurrentSimulation->GetQueue()->InsertEventInBuffer(this->OpenMP_index, i, timeDrivenPropagatedSpike[i][index]);
				}

				//Create a new TimeDrivenPropagatedSpike event
				PropagationStructure->IncrementEventSize(i, index);
				timeDrivenPropagatedSpike[i][index]=new TimeDrivenPropagatedSpike(timeDrivenPropagatedSpike[i][index]->GetTime(),i,PropagationStructure->GetEventSize(i, index));
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

void TimeDrivenInternalSpike::ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction){

	if(RealTimeRestriction<2){

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


		for (int i = 0; i<NumberOfOpenMPQueues; i++){
			for (int j=0; j<this->PropagationStructure->GetSize(i); j++){
				//Check if the each TimeDrivenPropagatedSpike event is empty.
				if(timeDrivenPropagatedSpike[i][j]->GetN_Elementes()>0){
					if (i == this->OpenMP_index){
						CurrentSimulation->GetQueue()->InsertEvent(i, timeDrivenPropagatedSpike[i][j]);
					}
					else{
						CurrentSimulation->GetQueue()->InsertEventInBuffer(this->OpenMP_index, i, timeDrivenPropagatedSpike[i][j]);
					}
				}else{
					delete timeDrivenPropagatedSpike[i][j];
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

	for (int i = 0; i<NumberOfOpenMPQueues; i++){
		for (int j=0; j<this->PropagationStructure->GetSize(i); j++){
			//Check if the each TimeDrivenPropagatedSpike event is empty.
			if(timeDrivenPropagatedSpike[i][j]->GetN_Elementes()>0){
				if (i == this->OpenMP_index){
					CurrentSimulation->GetQueue()->InsertEvent(i, timeDrivenPropagatedSpike[i][j]);
				}
				else{
					CurrentSimulation->GetQueue()->InsertEventInBuffer(this->OpenMP_index, i, timeDrivenPropagatedSpike[i][j]);
				}
			}else{
				delete timeDrivenPropagatedSpike[i][j];
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


//TimeDrivenInternalSpike::TimeDrivenInternalSpike(double NewTime, VectorNeuronState * NewState, Neuron ** NewNeurons, int NewOpenMP_index) : Spike(NewTime, NULL), State(NewState), Neurons(NewNeurons), OpenMP_index(NewOpenMP_index){
//}
//   		
//TimeDrivenInternalSpike::~TimeDrivenInternalSpike(){
//}
//
//void TimeDrivenInternalSpike::ProcessInternalSpikeEvent(Simulation * CurrentSimulation, int index){
//	Neuron * neuron = Neurons[index];
//	this->SetSource(neuron);
//	CurrentSimulation->IncrementTotalSpikeCounter( neuron->get_OpenMP_queue_index()); 
//
//	CurrentSimulation->WriteSpike(this);
//	CurrentSimulation->WriteState(neuron->GetVectorNeuronState()->GetLastUpdateTime(neuron->GetIndex_VectorNeuronState()), neuron);
//
//
//	// Generate the output activity
//	for (int i = 0; i<NumberOfOpenMPQueues; i++){
//		if (neuron->IsOutputConnected(i)){
//			PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + neuron->SynapseDelays[i][0], neuron, 0, i);
//			if(i==neuron->get_OpenMP_queue_index()){
//				CurrentSimulation->GetQueue()->InsertEvent(i,spike);
//			}else{
//				CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
//			}
//		}
//	}
//
//	if (neuron->GetInputNumberWithPostSynapticLearning()>0){
//		#ifdef _OPENMP 
//			#if	_OPENMP >= OPENMPVERSION30
//				#pragma omp task
//			#endif
//		#endif
//
//		{
//			for (int i = 0; i < neuron->GetInputNumberWithPostSynapticLearning(); ++i){
//				Interconnection * inter = neuron->GetInputConnectionWithPostSynapticLearningAt(i);
//				inter->GetWeightChange_withPost()->ApplyPostSynapticSpike(inter, this->time);
//			}
//		}
//
//	}
//}
//
//void TimeDrivenInternalSpike::ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction){
//
//	if(RealTimeRestriction<2){
//
//		//CPU write in this array if an internal spike must be generated.
//		bool * generateInternalSpike = State->getInternalSpike();
//
//		Neuron * Cell;
//
//		//We check if some neuron inside the model is monitored
//		if (State->Get_Is_Monitored()){
//			for (int t = 0; t<State->GetSizeState(); t++){
//				Cell = Neurons[t];
//				if (generateInternalSpike[t] == true){
//					ProcessInternalSpikeEvent(CurrentSimulation, t);
//				}
//				if (Cell->IsMonitored()){
//					CurrentSimulation->WriteState(this->GetTime(), Cell);
//				}
//			}
//		}
//		else{
//			for (int t = 0; t<State->GetSizeState(); t++){
//				if (generateInternalSpike[t] == true){
//					Cell = Neurons[t];
//					ProcessInternalSpikeEvent(CurrentSimulation, t);
//				}
//			}
//
//		}
//
//		#ifdef _OPENMP 
//			#if	_OPENMP >= OPENMPVERSION30
//				#pragma omp taskwait
//			#endif
//		#endif
//	}
//}
//
//void TimeDrivenInternalSpike::ProcessEvent(Simulation * CurrentSimulation){
//
//	//CPU write in this array if an internal spike must be generated.
//	bool * generateInternalSpike = State->getInternalSpike();
//
//	Neuron * Cell;
//
//	//We check if some neuron inside the model is monitored
//	if (State->Get_Is_Monitored()){
//		for (int t = 0; t<State->GetSizeState(); t++){
//			Cell = Neurons[t];
//			if (generateInternalSpike[t] == true){
//				ProcessInternalSpikeEvent(CurrentSimulation, t);
//			}
//			if (Cell->IsMonitored()){
//				CurrentSimulation->WriteState(this->GetTime(), Cell);
//			}
//		}
//	}
//	else{
//		for (int t = 0; t<State->GetSizeState(); t++){
//			if (generateInternalSpike[t] == true){
//				Cell = Neurons[t];
//				ProcessInternalSpikeEvent(CurrentSimulation, t);
//			}
//		}
//
//	}
//
//	#ifdef _OPENMP 
//		#if	_OPENMP >= OPENMPVERSION30
//			#pragma omp taskwait
//		#endif
//	#endif
//}
//   	
//void TimeDrivenInternalSpike::PrintType(){
//	cout<<"TimeDrivenInternalSpike"<<endl;
//}