/***************************************************************************
 *                           BDF1ad.cpp                                    *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
#include <math.h>
#include "../../include/integration_method/BDF1ad.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


BDF1ad::BDF1ad(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread):VariableStep("BDF1ad", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, N_CPU_thread, false, false),
e_min(0), e_max(0), h_min(0), h_max(0)
{	
	BDF=new BDF1vs(N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, N_CPU_thread);
}

BDF1ad::~BDF1ad(){
	delete BDF;
	free(PredictedNeuronState);
	free(ValidPrediction);
}
		
void BDF1ad::NextDifferentialEcuationValue(int index, TimeDrivenNeuronModel * Model, float * NeuronState, double elapsed_time, int CPU_thread_index){
	float * offset_PredictedNeuronState = PredictedNeuronState+(N_NeuronStateVariables*index);
	
	if(ValidPrediction[index]){
		memcpy(NeuronState, offset_PredictedNeuronState,sizeof(float)*N_NeuronStateVariables);
	}else{
		this->BDF->NextDifferentialEcuationValue(index, Model, NeuronState, elapsed_time, CPU_thread_index);
		memcpy(offset_PredictedNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
		ValidPrediction[index]=true;
	}


	bool stop=false;
	bool increment=false;
	bool decrement=false;

	while(!stop){
		stop=true;

		this->BDF->NextDifferentialEcuationValue(index, Model, offset_PredictedNeuronState, PredictedElapsedTime[index], CPU_thread_index);
		
		if ((PredictedElapsedTime[index]>h_min && (BDF->Epsilon[CPU_thread_index] > e_max))|| BDF->Epsilon[CPU_thread_index]!=BDF->Epsilon[CPU_thread_index]){
			stop=false;
			PredictedElapsedTime[index] *= 0.5;
			memcpy(offset_PredictedNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
			decrement=true;
			BDF->ReturnToOriginalState(index);
			if(BDF->Epsilon[CPU_thread_index]!=BDF->Epsilon[CPU_thread_index]){
				printf("ERROR2\n");
			}
		}
		else if (BDF->Epsilon[CPU_thread_index] < e_min && !decrement && !increment){
			stop=false;
			PredictedElapsedTime[index] *= 2;
			memcpy(offset_PredictedNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
			increment=true;
			BDF->ReturnToOriginalState(index);
		}

		if(PredictedElapsedTime[index]>h_max){
			PredictedElapsedTime[index]=h_max;
		}
		if(PredictedElapsedTime[index]<h_min){
			PredictedElapsedTime[index]=h_min;
		}
	}

}

ostream & BDF1ad::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	


void BDF1ad::InitializeStates(int N_neurons, float * initialization){
	BDF->InitializeStates(N_neurons, initialization);
	this->PredictedNeuronState=new float [N_neurons*N_NeuronStateVariables];
	this->ValidPrediction=new bool [N_neurons]();


	double elapsedTime=PredictedElapsedTime[0];
	free(PredictedElapsedTime);
	PredictedElapsedTime=new double[N_neurons];

	for(int i=0; i<N_neurons; i++){
		PredictedElapsedTime[i]=elapsedTime;
	}
}



void BDF1ad::loadParameter(FILE *fh, long * Currentline) throw (EDLUTFileException){
	this->PredictedElapsedTime=new double [1];

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%lf",PredictedElapsedTime)==1){
		if(PredictedElapsedTime[0]<=0.0){
////NEW CODE------------------------------------------------------------------------------
			throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
		}
	}else{
//NEW CODE------------------------------------------------------------------------------
		throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
	}

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%f",&h_max)==1){
		if(h_max<=0.0){
////NEW CODE------------------------------------------------------------------------------
			throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
		}
	}else{
//NEW CODE------------------------------------------------------------------------------
		throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
	}

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%f",&h_min)==1){
		if(h_min<=0.0){
////NEW CODE------------------------------------------------------------------------------
			throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
		}
	}else{
//NEW CODE------------------------------------------------------------------------------
		throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
	}

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%f",&e_max)==1){
		if(e_max<=0.0){
////NEW CODE------------------------------------------------------------------------------
			throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
		}
	}else{
//NEW CODE------------------------------------------------------------------------------
		throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
	}

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%f",&e_min)==1){
		if(e_min<=0.0){
////NEW CODE------------------------------------------------------------------------------
			throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
		}
	}else{
//NEW CODE------------------------------------------------------------------------------
		throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
	}

}

