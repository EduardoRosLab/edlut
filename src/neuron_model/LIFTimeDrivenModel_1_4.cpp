/***************************************************************************
 *                           LIFTimeDrivenModel_1_4.cpp                    *
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

#include "../../include/neuron_model/LIFTimeDrivenModel_1_4.h"
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

void LIFTimeDrivenModel_1_4::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
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
							skip_comments(fh,Currentline);

							if(fscanf(fh,"%f",&this->tampa)==1){
								skip_comments(fh,Currentline);

								if(fscanf(fh,"%f",&this->tnmda)==1){
									skip_comments(fh,Currentline);
									
									if(fscanf(fh,"%f",&this->tinh)==1){
										skip_comments(fh,Currentline);

										if(fscanf(fh,"%f",&this->tgj)==1){
											skip_comments(fh,Currentline);
											if(fscanf(fh,"%f",&this->tref)==1){
												skip_comments(fh,Currentline);

												if(fscanf(fh,"%f",&this->grest)==1){
													skip_comments(fh,Currentline);

													if(fscanf(fh,"%f",&this->fgj)==1){
														skip_comments(fh,Currentline);

														this->InitialState = (VectorNeuronState *) new VectorNeuronState(5, true);

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
				} else {
					throw EDLUTFileException(13,69,3,1,Currentline);
				}
			} else {
				throw EDLUTFileException(13,70,3,1,Currentline);
			}
		} else {
			throw EDLUTFileException(13,71,3,1,Currentline);
		}

		//INTEGRATION METHOD
		this->integrationMethod = LoadIntegrationMethod::loadIntegrationMethod((TimeDrivenNeuronModel *)this, fh, &Currentline, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
	}
}

void LIFTimeDrivenModel_1_4::SynapsisEffect(int index, Interconnection * InputConnection){

	switch (InputConnection->GetType()){
		case 0: {
			this->GetVectorNeuronState()->IncrementStateVariableAtCPU(index,N_DifferentialNeuronState,InputConnection->GetWeight());
			break;
		}case 1:{
			this->GetVectorNeuronState()->IncrementStateVariableAtCPU(index,N_DifferentialNeuronState+1,InputConnection->GetWeight());
			break;
		}case 2:{
			this->GetVectorNeuronState()->IncrementStateVariableAtCPU(index,N_DifferentialNeuronState+2,InputConnection->GetWeight());
			break;
		}case 3:{
			this->GetVectorNeuronState()->IncrementStateVariableAtCPU(index,N_DifferentialNeuronState+3,InputConnection->GetWeight());
			break;
		}default :{
			printf("ERROR: LIFTimeDrivenModel_1_4 only support four kind of input synapses \n");
		}
	}

}

LIFTimeDrivenModel_1_4::LIFTimeDrivenModel_1_4(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel(NeuronTypeID, NeuronModelID), eexc(0), einh(0), erest(0), vthr(0), cm(0), tampa(0), tnmda(0), tinh(0), tgj(0),
		tref(0), grest(0){
}

LIFTimeDrivenModel_1_4::~LIFTimeDrivenModel_1_4(void)
{
}

void LIFTimeDrivenModel_1_4::LoadNeuronModel() throw (EDLUTFileException){
	this->LoadNeuronModel(this->GetModelID()+".cfg");
}

VectorNeuronState * LIFTimeDrivenModel_1_4::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * LIFTimeDrivenModel_1_4::ProcessInputSpike(PropagatedSpike *  InputSpike){
	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(omp_get_thread_num(),InputSpike->GetTarget());

	Neuron * TargetCell = inter->GetTarget();

	// Add the effect of the input spike
	this->SynapsisEffect(inter->GetTarget()->GetIndex_VectorNeuronState(),inter);


	return 0;
}


InternalSpike * LIFTimeDrivenModel_1_4::ProcessInputSpike(Interconnection * inter, Neuron * target, double time){
		// Add the effect of the input spike
	this->SynapsisEffect(target->GetIndex_VectorNeuronState(),inter);

	return 0;
}


bool LIFTimeDrivenModel_1_4::UpdateState(int index, VectorNeuronState * State, double CurrentTime){

	
	bool * internalSpike=State->getInternalSpike();
	int Size=State->GetSizeState();
	double last_update;
	double elapsed_time;
	float elapsed_time_f;
	double last_spike;
	bool spike;
	float vm_cou;
	int i;
	int CPU_thread_index;

	float * NeuronState;
	//NeuronState[0] --> vm 
	//NeuronState[1] --> gampa 
	//NeuronState[2] --> gnmda 
	//NeuronState[3] --> ginh 
	//NeuronState[4] --> ggj 


	if(index==-1){
		for(int j=0; j<NumberOfOpenMPTasks-1; j++){
			#ifdef _OPENMP 
				#if	_OPENMP >= OPENMPVERSION30
					#pragma omp task firstprivate (j)
				#endif
			#endif
			{
				for (int i=LimitOfOpenMPTasks[j]; i< LimitOfOpenMPTasks[j+1]; i++){
					last_update = State->GetLastUpdateTime(i);
					elapsed_time = CurrentTime - last_update;
					elapsed_time_f=elapsed_time;
					State->AddElapsedTime(i,elapsed_time);
					last_spike = State->GetLastSpikeTime(i);

					NeuronState=State->GetStateVariableAt(i);
				
					spike = false;

					if (last_spike > this->tref) {
						this->integrationMethod->NextDifferentialEcuationValue(i, NeuronState, elapsed_time_f);
						vm_cou = NeuronState[0] + this->fgj * NeuronState[4];
						if (vm_cou > this->vthr){
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
			last_update = State->GetLastUpdateTime(i);
			elapsed_time = CurrentTime - last_update;
			elapsed_time_f=elapsed_time;
			State->AddElapsedTime(i,elapsed_time);
			last_spike = State->GetLastSpikeTime(i);

			NeuronState=State->GetStateVariableAt(i);
				
			spike = false;

			if (last_spike > this->tref) {
				this->integrationMethod->NextDifferentialEcuationValue(i, NeuronState, elapsed_time_f);
				vm_cou = NeuronState[0] + this->fgj * NeuronState[4];
				if (vm_cou > this->vthr){
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
		last_update = State->GetLastUpdateTime(index);
		elapsed_time = CurrentTime - last_update;
		elapsed_time_f=elapsed_time;
		State->AddElapsedTime(index,elapsed_time);
		last_spike = State->GetLastSpikeTime(index);

		NeuronState=State->GetStateVariableAt(index);
			
		spike = false;

		if (last_spike > this->tref) {
			this->integrationMethod->NextDifferentialEcuationValue(index, NeuronState, elapsed_time_f);
			vm_cou = NeuronState[0] + this->fgj * NeuronState[4];
			if (vm_cou > this->vthr){
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
//bool LIFTimeDrivenModel_1_4::UpdateState(int index, VectorNeuronState * State, double CurrentTime){
//
//	
//	bool * internalSpike=State->getInternalSpike();
//	int Size=State->GetSizeState();
//	double last_update;
//	double elapsed_time;
//	float elapsed_time_f;
//	double last_spike;
//	bool spike;
//	float vm_cou;
//	int i;
//	int CPU_thread_index;
//
//	float * NeuronState;
//	//NeuronState[0] --> vm 
//	//NeuronState[1] --> gampa 
//	//NeuronState[2] --> gnmda 
//	//NeuronState[3] --> ginh 
//	//NeuronState[4] --> ggj 
//
//
//	if(index==-1){
//		for(int j=0; j<NumberOfOpenMPTasks; j++){
//			#ifdef _OPENMP 
//				#if	_OPENMP >= OPENMPVERSION30
//					#pragma omp task if(j<(NumberOfOpenMPTasks-1)) firstprivate (j)
//				#endif
//			#endif
//			{
//				for (int i=LimitOfOpenMPTasks[j]; i< LimitOfOpenMPTasks[j+1]; i++){
//					last_update = State->GetLastUpdateTime(i);
//					elapsed_time = CurrentTime - last_update;
//					elapsed_time_f=elapsed_time;
//					State->AddElapsedTime(i,elapsed_time);
//					last_spike = State->GetLastSpikeTime(i);
//
//					NeuronState=State->GetStateVariableAt(i);
//				
//					spike = false;
//
//					if (last_spike > this->tref) {
//						this->integrationMethod->NextDifferentialEcuationValue(i, NeuronState, elapsed_time_f);
//						vm_cou = NeuronState[0] + this->fgj * NeuronState[4];
//						if (vm_cou > this->vthr){
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
//		#ifdef _OPENMP 
//			#if	_OPENMP >= OPENMPVERSION30
//				#pragma omp taskwait
//			#endif
//		#endif
//
//	}
//
//	else{
//		last_update = State->GetLastUpdateTime(index);
//		elapsed_time = CurrentTime - last_update;
//		elapsed_time_f=elapsed_time;
//		State->AddElapsedTime(index,elapsed_time);
//		last_spike = State->GetLastSpikeTime(index);
//
//		NeuronState=State->GetStateVariableAt(index);
//			
//		spike = false;
//
//		if (last_spike > this->tref) {
//			this->integrationMethod->NextDifferentialEcuationValue(index, NeuronState, elapsed_time_f);
//			vm_cou = NeuronState[0] + this->fgj * NeuronState[4];
//			if (vm_cou > this->vthr){
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
//	return false;
//}





ostream & LIFTimeDrivenModel_1_4::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model 1_4: " << this->GetModelID() << endl;

	out << "\tExc. Reversal Potential: " << this->eexc << "V\tInh. Reversal Potential: " << this->einh << "V\tResting potential: " << this->erest << "V" << endl;

	out << "\tFiring threshold: " << this->vthr << "V\tMembrane capacitance: " << this->cm << "nS\tAMPA Time Constant: " << this->tampa << "sNMDA Time Constant: " << this->tnmda << "s" << endl;

	out << "\tInhibitory time constant: " << this->tinh << "s\tGap junction time constant: " << this->tgj << "s\tRefractory Period: " << this->tref << "s\tResting Conductance: " << this->grest << "nS" << endl;

	return out;
}	



void LIFTimeDrivenModel_1_4::InitializeStates(int N_neurons){
	//Initialize neural state variables.
	float initialization[] = {erest,0.0f,0.0f,0.0f,0.0f};
	InitialState->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integrationMethod->InitializeStates(N_neurons, initialization);

	//Calculate number of OpenMP task and size of each one.
	CalculateTaskSizes(N_neurons, 1000);
}



void LIFTimeDrivenModel_1_4::EvaluateDifferentialEcuation(float * NeuronState, float * AuxNeuronState){
	float iampa = NeuronState[1]*(this->eexc-NeuronState[0]);
	float gnmdainf = 1.0f/(1.0f + ExponentialTable::GetResult(-62.0f*NeuronState[0])*(1.2f/3.57f));
	float inmda = NeuronState[2]*gnmdainf*(this->eexc-NeuronState[0]);
	float iinh = NeuronState[3]*(this->einh-NeuronState[0]);
	AuxNeuronState[0]=(iampa + inmda + iinh + this->grest* (this->erest-NeuronState[0]))*1.e-9f/this->cm;
}

void LIFTimeDrivenModel_1_4::EvaluateTimeDependentEcuation(float * NeuronState, float elapsed_time){
	//NeuronState[1]*= ExponentialTable::GetResult(-(elapsed_time/this->tampa));
	//NeuronState[2]*= ExponentialTable::GetResult(-(elapsed_time/this->tnmda));
	//NeuronState[3]*= ExponentialTable::GetResult(-(elapsed_time/this->tinh));
	//NeuronState[4]*= ExponentialTable::GetResult(-(elapsed_time/this->tgj));
	
	if(NeuronState[1]<1e-30){
		NeuronState[1]=0.0f;
	}else{
		NeuronState[1]*= ExponentialTable::GetResult(-(elapsed_time/this->tampa));
	}
	if(NeuronState[2]<1e-30){
		NeuronState[2]=0.0f;
	}else{
		NeuronState[2]*= ExponentialTable::GetResult(-(elapsed_time/this->tnmda));
	}
	if(NeuronState[3]<1e-30){
		NeuronState[3]=0.0f;
	}else{
		NeuronState[3]*= ExponentialTable::GetResult(-(elapsed_time/this->tinh));
	}
	if(NeuronState[4]<1e-30){
		NeuronState[4]=0.0f;
	}else{
		NeuronState[4]*= ExponentialTable::GetResult(-(elapsed_time/this->tgj));
	}
}

