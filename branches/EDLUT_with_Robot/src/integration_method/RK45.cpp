/***************************************************************************
 *                           RK45.cpp                                       *
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
#include "../../include/integration_method/RK45.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


RK45::RK45(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):FixedStep(NewModel, "RK45", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, false, false),
	a1(25.0f/216.0f), a2(0), a3(1408.0f/2565.0f), a4(2197.0f/4104.0f), a5(-0.2f),
	b1(16.0f/135.0f), b2(0), b3(6656.0f/12825.0f), b4(28561.0f/56430.0f), b5(-0.18f), b6(2.0f/55.0f),
	c20(0.25f), c21(0.25f),
	c30(0.375f), c31(0.09375f), c32(0.28125f),
	c40(12.0f/13.0f), c41(1932.0f/2197.0f), c42(-7200.0f/2197.0f), c43(7296.0f/2197.0f),
	c51(439.0f/216.0f), c52(-8.0f), c53(439.0f/216.0f), c54(-845.0f/4104.0f),
	c60(0.5f), c61(-8.0f/27.0f), c62(2), c63(-3544.0f/2565.0f), c64(1859.0f/4104.0f), c65(-0.275f)
{	
}

RK45::~RK45(){
}
		
void RK45::NextDifferentialEcuationValue(int index, float * NeuronState, float elapsed_time){
	float AuxNeuronState[MAX_VARIABLES];
	float AuxNeuronState1[MAX_VARIABLES];
	float AuxNeuronState2[MAX_VARIABLES];
	float AuxNeuronState3[MAX_VARIABLES];
	float AuxNeuronState4[MAX_VARIABLES];
	float AuxNeuronState5[MAX_VARIABLES];
	float AuxNeuronState6[MAX_VARIABLES];
	float x4[MAX_VARIABLES];

	//1st term
	model->EvaluateDifferentialEcuation(NeuronState, AuxNeuronState1);
	
	//2nd term
	memcpy(AuxNeuronState, NeuronState,sizeof(float)*N_NeuronStateVariables);
	for (int j=0; j<N_DifferentialNeuronState; j++){
		AuxNeuronState[j]= NeuronState[j] + c21*AuxNeuronState1[j]*elapsed_time;
	}

	model->EvaluateTimeDependentEcuation(AuxNeuronState, c20*elapsed_time);
	model->EvaluateDifferentialEcuation(AuxNeuronState, AuxNeuronState2);

	//3rd term
	for (int j=0; j<N_DifferentialNeuronState; j++){
		AuxNeuronState[j]=NeuronState[j] + (c31*AuxNeuronState1[j] + c32*AuxNeuronState2[j])*elapsed_time;
	}

	model->EvaluateTimeDependentEcuation(AuxNeuronState, c30*elapsed_time);
	model->EvaluateDifferentialEcuation(AuxNeuronState, AuxNeuronState3);

	//4rd term
	for (int j=0; j<N_DifferentialNeuronState; j++){
		AuxNeuronState[j]=NeuronState[j] + (c41*AuxNeuronState1[j] + c42*AuxNeuronState2[j] + c43*AuxNeuronState3[j])*elapsed_time;
	}

	model->EvaluateTimeDependentEcuation(AuxNeuronState, c40*elapsed_time);
	model->EvaluateDifferentialEcuation(AuxNeuronState, AuxNeuronState4);

	//5rd term
	for (int j=0; j<N_DifferentialNeuronState; j++){
		AuxNeuronState[j]=NeuronState[j] + (c51*AuxNeuronState1[j] + c52*AuxNeuronState2[j] + c53*AuxNeuronState3[j] + c54*AuxNeuronState4[j])*elapsed_time;
	}

	model->EvaluateTimeDependentEcuation(AuxNeuronState, elapsed_time);
	model->EvaluateDifferentialEcuation(AuxNeuronState, AuxNeuronState5);

	//6rd term
	for (int j=0; j<N_DifferentialNeuronState; j++){
		AuxNeuronState[j]=NeuronState[j] + (c61*AuxNeuronState1[j] + c62*AuxNeuronState2[j] + c63*AuxNeuronState3[j] + c64*AuxNeuronState4[j] + c65*AuxNeuronState5[j])*elapsed_time;
	}

	model->EvaluateTimeDependentEcuation(AuxNeuronState, c60*elapsed_time);
	model->EvaluateDifferentialEcuation(AuxNeuronState, AuxNeuronState6);

	for (int j=0; j<N_DifferentialNeuronState; j++){
		x4[j]=NeuronState[j] + (a1*AuxNeuronState1[j] + a3*AuxNeuronState3[j] + a4*AuxNeuronState4[j] + a5*AuxNeuronState5[j])*elapsed_time;
		NeuronState[j]+=(b1*AuxNeuronState1[j] + b3*AuxNeuronState3[j] + b4*AuxNeuronState4[j] + b5*AuxNeuronState5[j] + b6*AuxNeuronState6[j])*elapsed_time;
		epsilon[index]+=fabs(NeuronState[j]-x4[j]);
	}


	model->EvaluateTimeDependentEcuation(NeuronState, elapsed_time);
}

ostream & RK45::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	

void RK45::InitializeStates(int N_neurons, float * initialization){
	this->epsilon=new float(N_neurons);
}
