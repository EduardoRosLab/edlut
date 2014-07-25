/***************************************************************************
 *                           RK45ad.cpp                                    *
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
#include "../../include/integration_method/RK45ad.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


RK45ad::RK45ad(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):VariableStep(NewModel, "RK45ad", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, false, false),
e_min(0), e_max(0), h_min(0), h_max(0)
{	
	RK=new RK45(NewModel, N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState);
}

RK45ad::~RK45ad(){
	delete RK;
	delete [] PredictedNeuronState;
	delete [] ValidPrediction;
}
		
void RK45ad::NextDifferentialEcuationValue(int index, float * NeuronState, float elapsed_time){
	float * offset_PredictedNeuronState = PredictedNeuronState+(N_NeuronStateVariables*index);
	


	if(ValidPrediction[index]){
		memcpy(NeuronState, offset_PredictedNeuronState,sizeof(float)*N_NeuronStateVariables);
	}else{
		this->RK->NextDifferentialEcuationValue(index, NeuronState, elapsed_time);
		memcpy(offset_PredictedNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
	}

	bool stop=false;
	bool increment=false;
	bool decrement=false;

	while(!stop){
		stop=true;
		//if(PredictedElapsedTime[index]<hmin){
		//	PredictedElapsedTime[index]=hmin;
		//}


		this->RK->NextDifferentialEcuationValue(index, offset_PredictedNeuronState, PredictedElapsedTime[index]);

		if (RK->epsilon[index] < e_min && !decrement){
			//stop=false;
			PredictedElapsedTime[index] *= 2;
			memcpy(offset_PredictedNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
			increment=true;
		}else if ( (RK->epsilon[index] > e_max  && !increment)|| RK->epsilon[index]!=RK->epsilon[index]){
			stop=false;
			PredictedElapsedTime[index] *= 0.5;
			memcpy(offset_PredictedNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
			decrement=true;
		}

		if(PredictedElapsedTime[index]>h_max){
			PredictedElapsedTime[index]=h_max;
		}
	}

}

ostream & RK45ad::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	


void RK45ad::InitializeStates(int N_neurons, float * initialization){
	this->PredictedNeuronState=new float [N_neurons*N_NeuronStateVariables];

	this->ValidPrediction=new bool [N_neurons]();


	float elapsedTime=PredictedElapsedTime[0];
	delete [] PredictedElapsedTime;
	PredictedElapsedTime=new double[N_neurons];

	for(int i=0; i<N_neurons; i++){
		PredictedElapsedTime[i]=elapsedTime;
	}
}



void RK45ad::loadParameter(FILE *fh, long * Currentline) throw (EDLUTFileException){
	this->PredictedElapsedTime=new double [1];

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%f",PredictedElapsedTime)==1){
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