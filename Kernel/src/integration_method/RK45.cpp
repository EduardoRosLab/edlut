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


RK45::RK45(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread):FixedStep("RK45", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, N_CPU_thread, false, false),
	a1(25.0f/216.0f), a2(0), a3(1408.0f/2565.0f), a4(2197.0f/4104.0f), a5(-0.2f),
	b1(16.0f/135.0f), b2(0), b3(6656.0f/12825.0f), b4(28561.0f/56430.0f), b5(-0.18f), b6(2.0f/55.0f),
	c20(0.25f), c21(0.25f),
	c30(0.375f), c31(0.09375f), c32(0.28125f),
	c40(12.0f/13.0f), c41(1932.0f/2197.0f), c42(-7200.0f/2197.0f), c43(7296.0f/2197.0f),
	c51(439.0f/216.0f), c52(-8.0f), c53(439.0f/216.0f), c54(-845.0f/4104.0f),
	c60(0.5f), c61(-8.0f/27.0f), c62(2), c63(-3544.0f/2565.0f), c64(1859.0f/4104.0f), c65(-0.275f)
{	
	AuxNeuronState = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState1 = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState2 = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState3 = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState4 = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState5 = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState6 = new float [N_NeuronStateVariables*N_CPU_thread];
	x4 = new float [N_NeuronStateVariables*N_CPU_thread];
	epsilon = new float [N_CPU_thread];
}

RK45::~RK45(){
	delete [] AuxNeuronState;
	delete [] AuxNeuronState1;
	delete [] AuxNeuronState2;
	delete [] AuxNeuronState3;
	delete [] AuxNeuronState4;
	delete [] AuxNeuronState5;
	delete [] AuxNeuronState6;
	delete [] x4;
	delete [] epsilon;
}
		
void RK45::NextDifferentialEcuationValue(int index, TimeDrivenNeuronModel * Model, float * NeuronState, float elapsed_time, int CPU_thread_index){
	float * offset_AuxNeuronState = AuxNeuronState+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState1 = AuxNeuronState1+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState2 = AuxNeuronState2+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState3 = AuxNeuronState3+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState4 = AuxNeuronState4+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState5 = AuxNeuronState3+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState6 = AuxNeuronState4+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_x4 = x4+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_epsilon = epsilon+(CPU_thread_index);


	//1st term
	Model->EvaluateDifferentialEcuation(NeuronState, offset_AuxNeuronState1);
	
	//2nd term
	memcpy(offset_AuxNeuronState, NeuronState,sizeof(float)*N_NeuronStateVariables);
	for (int j=0; j<N_DifferentialNeuronState; j++){
		offset_AuxNeuronState[j]= NeuronState[j] + c21*offset_AuxNeuronState1[j]*elapsed_time;
	}

	Model->EvaluateTimeDependentEcuation(offset_AuxNeuronState, c20*elapsed_time);
	Model->EvaluateDifferentialEcuation(offset_AuxNeuronState, offset_AuxNeuronState2);

	//3rd term
	for (int j=0; j<N_DifferentialNeuronState; j++){
		offset_AuxNeuronState[j]=NeuronState[j] + (c31*offset_AuxNeuronState1[j] + c32*offset_AuxNeuronState2[j])*elapsed_time;
	}

	Model->EvaluateTimeDependentEcuation(offset_AuxNeuronState, c30*elapsed_time);
	Model->EvaluateDifferentialEcuation(offset_AuxNeuronState, offset_AuxNeuronState3);

	//4rd term
	for (int j=0; j<N_DifferentialNeuronState; j++){
		offset_AuxNeuronState[j]=NeuronState[j] + (c41*offset_AuxNeuronState1[j] + c42*offset_AuxNeuronState2[j] + c43*offset_AuxNeuronState3[j])*elapsed_time;
	}

	Model->EvaluateTimeDependentEcuation(offset_AuxNeuronState, c40*elapsed_time);
	Model->EvaluateDifferentialEcuation(offset_AuxNeuronState, offset_AuxNeuronState4);

	//5rd term
	for (int j=0; j<N_DifferentialNeuronState; j++){
		offset_AuxNeuronState[j]=NeuronState[j] + (c51*offset_AuxNeuronState1[j] + c52*offset_AuxNeuronState2[j] + c53*offset_AuxNeuronState3[j] + c54*offset_AuxNeuronState4[j])*elapsed_time;
	}

	Model->EvaluateTimeDependentEcuation(offset_AuxNeuronState, elapsed_time);
	Model->EvaluateDifferentialEcuation(offset_AuxNeuronState, offset_AuxNeuronState5);

	//6rd term
	for (int j=0; j<N_DifferentialNeuronState; j++){
		offset_AuxNeuronState[j]=NeuronState[j] + (c61*offset_AuxNeuronState1[j] + c62*offset_AuxNeuronState2[j] + c63*offset_AuxNeuronState3[j] + c64*offset_AuxNeuronState4[j] + c65*offset_AuxNeuronState5[j])*elapsed_time;
	}

	Model->EvaluateTimeDependentEcuation(offset_AuxNeuronState, c60*elapsed_time);
	Model->EvaluateDifferentialEcuation(offset_AuxNeuronState, offset_AuxNeuronState6);

	offset_epsilon[CPU_thread_index]=0;
	float aux;
	for (int j=0; j<N_DifferentialNeuronState; j++){
		offset_x4[j]=NeuronState[j] + (a1*offset_AuxNeuronState1[j] + a3*offset_AuxNeuronState3[j] + a4*offset_AuxNeuronState4[j] + a5*offset_AuxNeuronState5[j])*elapsed_time;
		NeuronState[j]+=(b1*offset_AuxNeuronState1[j] + b3*offset_AuxNeuronState3[j] + b4*offset_AuxNeuronState4[j] + b5*offset_AuxNeuronState5[j] + b6*offset_AuxNeuronState6[j])*elapsed_time;
		//aux=fabs(NeuronState[j]-offset_x4[j]);
		//if(offset_epsilon[CPU_thread_index]<aux){
		//	offset_epsilon[CPU_thread_index]=aux;
		//}
		offset_epsilon[CPU_thread_index]+=fabs(NeuronState[j]-offset_x4[j]);
	}


	Model->EvaluateTimeDependentEcuation(NeuronState, elapsed_time);
}

ostream & RK45::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	
