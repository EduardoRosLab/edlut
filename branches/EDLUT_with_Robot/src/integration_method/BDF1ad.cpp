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


BDF1ad::BDF1ad(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):VariableStep(NewModel, "BDF1ad", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, false, false),
e_min(0), e_max(0), h_min(0), h_max(0)
{	
	BDF=new BDF1vs(NewModel, N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState);
}

BDF1ad::~BDF1ad(){
	delete BDF;
	delete [] PredictedNeuronState;
	delete [] ValidPrediction;
	delete [] NextStepPredictedElapsedTime;
}
		
void BDF1ad::NextDifferentialEcuationValue(int index, float * NeuronState, float elapsed_time){
	float tolerance=e_max;
	
	float * offset_PredictedNeuronState = PredictedNeuronState+(N_NeuronStateVariables*index);
	
	if(ValidPrediction[index]){
		memcpy(NeuronState, offset_PredictedNeuronState,sizeof(float)*N_NeuronStateVariables);
	}else{
		this->BDF->NextDifferentialEcuationValue(index, NeuronState, elapsed_time);
		memcpy(offset_PredictedNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
		ValidPrediction[index]=true;
		NextStepPredictedElapsedTime[index]=h_min;
	}


	bool stop=false;
	float second_derivative;

	while(!stop){
		PredictedElapsedTime[index]=NextStepPredictedElapsedTime[index];
		stop=true;
		this->BDF->NextDifferentialEcuationValue(index, offset_PredictedNeuronState, PredictedElapsedTime[index]);
		
		second_derivative=0.0f;
		float second_derivative2=0.0f;
		for(int i=0; i<this->N_DifferentialNeuronState; i++){
			second_derivative+=fabs(this->BDF->D[index*N_DifferentialNeuronState + i] - this->BDF->OriginalD[index*N_DifferentialNeuronState + i]);
			second_derivative2+=this->BDF->D[index*N_DifferentialNeuronState + i];
		}
		second_derivative/=PredictedElapsedTime[index];


		if((second_derivative>=tolerance && NextStepPredictedElapsedTime[index]>h_min) || second_derivative2!=second_derivative2 ){
			NextStepPredictedElapsedTime[index]*=0.5f;
			BDF->ReturnToOriginalState(index);
			memcpy(offset_PredictedNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
			stop=false;
		}else{
			float ratio=0.0;
			if(second_derivative>= (tolerance*0.5f)){
				ratio=(tolerance*0.5f)/second_derivative;
				ratio=pow(ratio,0.333f);
				if(ratio<0.5){
					ratio=0.5;
				}else if(ratio>0.9){
					ratio=0.9;
				}
			}else{
				if(second_derivative>= (tolerance*0.0625f)){
					ratio=1.0f;
				}else{
					ratio=2.0f;
				}
			}
			NextStepPredictedElapsedTime[index]*=ratio;
		}

		if(NextStepPredictedElapsedTime[index]<h_min && second_derivative2==second_derivative2 ){
			NextStepPredictedElapsedTime[index]=h_min;
		}
		if(NextStepPredictedElapsedTime[index]>h_max){
			NextStepPredictedElapsedTime[index]=h_max;
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
	memset(ValidPrediction, 0, N_neurons);


	double elapsedTime=PredictedElapsedTime[0];
	delete [] PredictedElapsedTime;
	PredictedElapsedTime=new double[N_neurons];
	NextStepPredictedElapsedTime=new double[N_neurons];

	for(int i=0; i<N_neurons; i++){
		PredictedElapsedTime[i]=elapsedTime;
		NextStepPredictedElapsedTime[i]=elapsedTime;
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

