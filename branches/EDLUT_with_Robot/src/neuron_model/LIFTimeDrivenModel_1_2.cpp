/***************************************************************************
 *                           LIFTimeDrivenModel_1_2.cpp                    *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@atc.ugr.es         *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/LIFTimeDrivenModel_1_2.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/openmp/openmp.h"

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"


void LIFTimeDrivenModel_1_2::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
		if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%f",&this->eexc)==1){
			skip_comments(fh,Currentline);

			if (fscanf(fh,"%f",&this->einh)==1){
				skip_comments(fh,Currentline);

				if(fscanf(fh,"%f",&this->erest)==1){
					skip_comments(fh,Currentline);

					if(fscanf(fh,"%f",&this->vthr)==1){
						skip_comments(fh,Currentline);

						if(fscanf(fh,"%f",&this->cm)==1){
							inv_cm=1.0f/cm;
							skip_comments(fh,Currentline);

							if(fscanf(fh,"%f",&this->texc)==1){
								inv_texc=1.0f/texc;
								skip_comments(fh,Currentline);

								if(fscanf(fh,"%f",&this->tinh)==1){
									inv_tinh=1.0f/tinh;
									skip_comments(fh,Currentline);

									if(fscanf(fh,"%f",&this->tref)==1){
										skip_comments(fh,Currentline);

										if(fscanf(fh,"%f",&this->grest)==1){
											skip_comments(fh,Currentline);

											this->InitialState = (VectorNeuronState *) new VectorNeuronState(3, true);

										} else {
											throw EDLUTFileException(13,60,3,1,Currentline);
										}
									} else {
										throw EDLUTFileException(13,61,3,1,Currentline);
									}
								} else {
									throw EDLUTFileException(13,62,3,1,Currentline);
								}
							} else {
								throw EDLUTFileException(13,63,3,1,Currentline);
							}
						} else {
							throw EDLUTFileException(13,64,3,1,Currentline);
						}
					} else {
						throw EDLUTFileException(13,65,3,1,Currentline);
					}
				} else {
					throw EDLUTFileException(13,66,3,1,Currentline);
				}
			} else {
				throw EDLUTFileException(13,67,3,1,Currentline);
			}
		} else {
			throw EDLUTFileException(13,68,3,1,Currentline);
		}
	
		//INTEGRATION METHOD
		this->integrationMethod = LoadIntegrationMethod::loadIntegrationMethod((TimeDrivenNeuronModel *)this, fh, &Currentline, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);

	}
}

void LIFTimeDrivenModel_1_2::SynapsisEffect(int index, Interconnection * InputConnection){
	switch (InputConnection->GetType()){
		case 0: {
			this->GetVectorNeuronState()->IncrementStateVariableAtCPU(index,N_DifferentialNeuronState,1e-9f*InputConnection->GetWeight());
			break;
		}case 1:{
			this->GetVectorNeuronState()->IncrementStateVariableAtCPU(index,N_DifferentialNeuronState+1,1e-9f*InputConnection->GetWeight());
			break;
		}default :{
			printf("ERROR: LIFTimeDrivenModel_1_2 only support two kind of input synapses \n");
		}
	}
}

LIFTimeDrivenModel_1_2::LIFTimeDrivenModel_1_2(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel(NeuronTypeID, NeuronModelID), eexc(0), einh(0), erest(0), vthr(0), cm(0), texc(0), tinh(0),
		tref(0), grest(0){
}

LIFTimeDrivenModel_1_2::~LIFTimeDrivenModel_1_2(void)
{
}

void LIFTimeDrivenModel_1_2::LoadNeuronModel() throw (EDLUTFileException){
	this->LoadNeuronModel(this->GetModelID()+".cfg");
}

VectorNeuronState * LIFTimeDrivenModel_1_2::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * LIFTimeDrivenModel_1_2::ProcessInputSpike(PropagatedSpike *  InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(omp_get_thread_num(),InputSpike->GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	// Add the effect of the input spike
	this->SynapsisEffect(TargetCell->GetIndex_VectorNeuronState(),inter);

	return 0;
}


InternalSpike * LIFTimeDrivenModel_1_2::ProcessInputSpike(Interconnection * inter, Neuron * target, double time){
	// Add the effect of the input spike
	this->SynapsisEffect(target->GetIndex_VectorNeuronState(),inter);

	return 0;
}






bool LIFTimeDrivenModel_1_2::UpdateState(int index, VectorNeuronState * State, double CurrentTime){

	
	bool * internalSpike=State->getInternalSpike();


	//float * NeuronState;
	//NeuronState[0] --> vm 
	//NeuronState[1] --> gexc 
	//NeuronState[2] --> ginh 



	if(index==-1){

		for(int j=0; j<NumberOfOpenMPTasks-1; j++){
			#ifdef _OPENMP 
				#if	_OPENMP >= OPENMPVERSION30
					#pragma omp task firstprivate (j) shared(internalSpike, State, CurrentTime) 
				#endif
			#endif
			{
				for (int i=LimitOfOpenMPTasks[j]; i< LimitOfOpenMPTasks[j+1]; i++){
					double last_update = State->GetLastUpdateTime(i);
					double elapsed_time = CurrentTime - last_update;
					float elapsed_time_f=elapsed_time;
					State->AddElapsedTime(i,elapsed_time);
					double last_spike = State->GetLastSpikeTime(i);

					float * NeuronState=State->GetStateVariableAt(i);
				
					bool spike = false;

					if (last_spike > this->tref) {
						this->integrationMethod->NextDifferentialEcuationValue(i, NeuronState, elapsed_time_f);
						if (NeuronState[0] > this->vthr){
							State->NewFiredSpike(i);
							spike = true;
							NeuronState[0] = this->erest;
							this->integrationMethod->resetState(i);
						}
					}else{
						EvaluateTimeDependentEcuation(NeuronState, elapsed_time_f);
					}

					internalSpike[i]=spike;

					State->SetLastUpdateTime(i,CurrentTime);
				}
			}
		}
		
		for (int i=LimitOfOpenMPTasks[NumberOfOpenMPTasks-1]; i< LimitOfOpenMPTasks[NumberOfOpenMPTasks]; i++){
			double last_update = State->GetLastUpdateTime(i);
			double elapsed_time = CurrentTime - last_update;
			float elapsed_time_f=elapsed_time;
			State->AddElapsedTime(i,elapsed_time);
			double last_spike = State->GetLastSpikeTime(i);

			float * NeuronState=State->GetStateVariableAt(i);
				
			bool spike = false;

			if (last_spike > this->tref) {
				this->integrationMethod->NextDifferentialEcuationValue(i, NeuronState, elapsed_time_f);
				if (NeuronState[0] > this->vthr){
					State->NewFiredSpike(i);
					spike = true;
					NeuronState[0] = this->erest;
					this->integrationMethod->resetState(i);
				}
			}else{
				EvaluateTimeDependentEcuation(NeuronState, elapsed_time_f);
			}

			internalSpike[i]=spike;

			State->SetLastUpdateTime(i,CurrentTime);
		}

		#ifdef _OPENMP 
			#if	_OPENMP >= OPENMPVERSION30
				#pragma omp taskwait
			#endif
		#endif
	}

	else{
		double last_update = State->GetLastUpdateTime(index);
		double elapsed_time = CurrentTime - last_update;
		float elapsed_time_f=elapsed_time;
		State->AddElapsedTime(index,elapsed_time);
		double last_spike = State->GetLastSpikeTime(index);

		float * NeuronState=State->GetStateVariableAt(index);
			
		bool spike = false;

		if (last_spike > this->tref) {
			this->integrationMethod->NextDifferentialEcuationValue(index, NeuronState, elapsed_time_f);
			if (NeuronState[0] > this->vthr){
				State->NewFiredSpike(index);
				spike = true;
				NeuronState[0] = this->erest;
				this->integrationMethod->resetState(index);
			}
		}else{
			EvaluateTimeDependentEcuation(NeuronState, elapsed_time_f);
		}

		internalSpike[index]=spike;

		State->SetLastUpdateTime(index,CurrentTime);
	}

	return false;
}

//
//bool LIFTimeDrivenModel_1_2::UpdateState(int index, VectorNeuronState * State, double CurrentTime){
//
//	
//	bool * internalSpike=State->getInternalSpike();
//	int Size=State->GetSizeState();
//
//	//NeuronState[0] --> vm 
//	//NeuronState[1] --> gexc 
//	//NeuronState[2] --> ginh 
//
//
//
//	if(index==-1){
//
//		for(int j=0; j<NumberOfOpenMPTasks; j++){
//			#ifdef _OPENMP 
//				#if	_OPENMP >= OPENMPVERSION30
//					#pragma omp task if (j<(NumberOfOpenMPTasks-1)) firstprivate (j) shared (internalSpike, Size, State, CurrentTime)
//				#endif
//			#endif
//			{
//				for (int i=LimitOfOpenMPTasks[j]; i< LimitOfOpenMPTasks[j+1]; i++){
//					double last_update = State->GetLastUpdateTime(i);
//					double elapsed_time = CurrentTime - last_update;
//					float elapsed_time_f=elapsed_time;
//					State->AddElapsedTime(i,elapsed_time);
//					double last_spike = State->GetLastSpikeTime(i);
//
//					float * NeuronState=State->GetStateVariableAt(i);
//				
//					bool spike = false;
//
//					if (last_spike > this->tref) {
//						this->integrationMethod->NextDifferentialEcuationValue(i, NeuronState, elapsed_time_f);
//						if (NeuronState[0] > this->vthr){
//							State->NewFiredSpike(i);
//							spike = true;
//							NeuronState[0] = this->erest;
//							this->integrationMethod->resetState(i);
//						}
//					}else{
//						EvaluateTimeDependentEcuation(NeuronState, elapsed_time_f);
//					}
//
//					internalSpike[i]=spike;
//
//					State->SetLastUpdateTime(i,CurrentTime);
//				}
//			}
//		}
//		
//		#ifdef _OPENMP 
//			#if	_OPENMP >= OPENMPVERSION30
//				#pragma omp taskwait
//			#endif
//		#endif
//	}
//
//	else{
//		double last_update = State->GetLastUpdateTime(index);
//		double elapsed_time = CurrentTime - last_update;
//		float elapsed_time_f=elapsed_time;
//		State->AddElapsedTime(index,elapsed_time);
//		double last_spike = State->GetLastSpikeTime(index);
//
//		float * NeuronState=State->GetStateVariableAt(index);
//			
//		bool spike = false;
//
//		if (last_spike > this->tref) {
//			this->integrationMethod->NextDifferentialEcuationValue(index, NeuronState, elapsed_time_f);
//			if (NeuronState[0] > this->vthr){
//				State->NewFiredSpike(index);
//				spike = true;
//				NeuronState[0] = this->erest;
//				this->integrationMethod->resetState(index);
//			}
//		}else{
//			EvaluateTimeDependentEcuation(NeuronState, elapsed_time_f);
//		}
//
//		internalSpike[index]=spike;
//
//
//
//		State->SetLastUpdateTime(index,CurrentTime);
//	}
//
//	return false;
//	
//}



ostream & LIFTimeDrivenModel_1_2::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model: " << this->GetModelID() << endl;

	out << "\tExc. Reversal Potential: " << this->eexc << "V\tInh. Reversal Potential: " << this->einh << "V\tResting potential: " << this->erest << "V" << endl;

	out << "\tFiring threshold: " << this->vthr << "V\tMembrane capacitance: " << this->cm << "nS\tExcitatory Time Constant: " << this->texc << "s" << endl;

	out << "\tInhibitory time constant: " << this->tinh << "s\tRefractory Period: " << this->tref << "s\tResting Conductance: " << this->grest << "nS" << endl;

	return out;
}	



void LIFTimeDrivenModel_1_2::InitializeStates(int N_neurons){
	//Initialize neural state variables.
	float initialization[] = {erest,0.0f,0.0f};
	InitialState->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integrationMethod->InitializeStates(N_neurons, initialization);


	//Calculate number of OpenMP task and size of each one.
	CalculateTaskSizes(N_neurons, 1000);
}



void LIFTimeDrivenModel_1_2::EvaluateDifferentialEcuation(float * NeuronState, float * AuxNeuronState){
	AuxNeuronState[0]=(NeuronState[1] * (this->eexc - NeuronState[0]) + NeuronState[2] * (this->einh - NeuronState[0]) + grest * (this->erest - NeuronState[0]))*this->inv_cm;
}

void LIFTimeDrivenModel_1_2::EvaluateTimeDependentEcuation(float * NeuronState, float elapsed_time){
	//NeuronState[1]*= ExponentialTable::GetResult(-(elapsed_time*this->inv_texc));
	//NeuronState[2]*= ExponentialTable::GetResult(-(elapsed_time*this->inv_tinh));

	float limit=1e-20;
	
	if(NeuronState[1]<limit){
		NeuronState[1]=0.0f;
	}else{
		NeuronState[1]*= ExponentialTable::GetResult(-(elapsed_time*this->inv_texc));
	}
	if(NeuronState[2]<limit){
		NeuronState[2]=0.0f;
	}else{
		NeuronState[2]*= ExponentialTable::GetResult(-(elapsed_time*this->inv_tinh));
	}	
}




