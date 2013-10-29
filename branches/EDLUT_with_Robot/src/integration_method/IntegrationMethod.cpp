/***************************************************************************
 *                           IntegrationMethod.cpp                         *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#include "../../include/integration_method/IntegrationMethod.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"

//#include "../../include/parallel_function.h"


IntegrationMethod::IntegrationMethod(string integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState,int N_timeDependentNeuronState, int N_CPU_thread, bool jacobian, bool inverse):IntegrationMethodType(integrationMethodType), N_NeuronStateVariables(N_neuronStateVariables), N_DifferentialNeuronState(N_differentialNeuronState), N_TimeDependentNeuronState(N_timeDependentNeuronState), N_CPU_Thread(N_CPU_thread){
	if(jacobian){
		JacAuxNeuronState = (float **)new float *[N_CPU_thread];
		JacAuxNeuronState_pos = (float **)new float *[N_CPU_thread];
		JacAuxNeuronState_neg = (float **)new float *[N_CPU_thread];
		for(int i=0; i<N_CPU_thread; i++){
			JacAuxNeuronState[i] = new float [N_NeuronStateVariables]();
			JacAuxNeuronState_pos[i] = new float [N_NeuronStateVariables]();
			JacAuxNeuronState_neg[i] = new float [N_NeuronStateVariables]();
		}
	}else{
		JacAuxNeuronState=0;
		JacAuxNeuronState_pos=0;
		JacAuxNeuronState_neg=0;
	}

}

IntegrationMethod::~IntegrationMethod(){
	if(JacAuxNeuronState!=0){
		for(int i=0; i<N_CPU_Thread; i++){
			delete JacAuxNeuronState[i];
		}
		delete [] JacAuxNeuronState;
	}

	if(JacAuxNeuronState_pos!=0){
		for(int i=0; i<N_CPU_Thread; i++){
			delete JacAuxNeuronState_pos[i];
		}
		delete [] JacAuxNeuronState_pos;
	}

	if(JacAuxNeuronState_neg!=0){
		for(int i=0; i<N_CPU_Thread; i++){
			delete JacAuxNeuronState_neg[i];
		}
		delete [] JacAuxNeuronState_neg;
	}

	//delete [] PredictedElapsedTime;
}

string IntegrationMethod::GetType(){
	return this->IntegrationMethodType;
}
		
void IntegrationMethod::Jacobian(TimeDrivenNeuronModel * Model, float * NeuronState, float * jacnum, int CPU_thread_index){
	float epsi=9.5367431640625e-7;
	float * offset_JacAuxNeuronState = JacAuxNeuronState[CPU_thread_index];
	float * offset_JacAuxNeuronState_pos = JacAuxNeuronState_pos[CPU_thread_index];
	float * offset_JacAuxNeuronState_neg = JacAuxNeuronState_neg[CPU_thread_index];

	for (int j=0; j<N_DifferentialNeuronState; j++){
		memcpy(offset_JacAuxNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
		offset_JacAuxNeuronState[j]+=epsi;
		Model->EvaluateDifferentialEcuation(offset_JacAuxNeuronState, offset_JacAuxNeuronState_pos);

		offset_JacAuxNeuronState[j]-=2*epsi;
		Model->EvaluateDifferentialEcuation(offset_JacAuxNeuronState, offset_JacAuxNeuronState_neg);

		for(int z=0; z<N_DifferentialNeuronState; z++){
			jacnum[z*N_DifferentialNeuronState+j]=(offset_JacAuxNeuronState_pos[z]-offset_JacAuxNeuronState_neg[z])/(2*epsi);
		}
	} 
}

//void IntegrationMethod::Jacobian(TimeDrivenNeuronModel * Model, float * NeuronState, float * jacnum, int CPU_thread_index, float elapsed_time){
//	float epsi=elapsed_time * 0.1f;
	//float * offset_JacAuxNeuronState = JacAuxNeuronState[CPU_thread_index];
	//float * offset_JacAuxNeuronState_pos = JacAuxNeuronState_pos[CPU_thread_index];
	//float * offset_JacAuxNeuronState_neg = JacAuxNeuronState_neg[CPU_thread_index];
//	for (int j=0; j<N_DifferentialNeuronState; j++){
//		memcpy(offset_JacAuxNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
//		offset_AuxNeuronState[j]+=epsi;
//		Model->EvaluateDifferentialEcuation(offset_JacAuxNeuronState, offset_JacAuxNeuronState_pos);
//
//		offset_JacAuxNeuronState[j]-=2*epsi;
//		Model->EvaluateDifferentialEcuation(offset_JacAuxNeuronState, offset_JacAuxNeuronState_neg);
//
//		for(int z=0; z<N_DifferentialNeuronState; z++){
//			jacnum[z*N_DifferentialNeuronState+j]=(offset_JacAuxNeuronState_pos[z]-offset_JacAuxNeuronState_neg[z])/(2*epsi);
//		}
//	} 
//}

void IntegrationMethod::Jacobian(TimeDrivenNeuronModel * Model, float * NeuronState, float * jacnum, int CPU_thread_index, float elapsed_time){
	float epsi=elapsed_time * 0.1f;
	float inv_epsi=1.0f/epsi;
	float * offset_JacAuxNeuronState = JacAuxNeuronState[CPU_thread_index];
	float * offset_JacAuxNeuronState_pos = JacAuxNeuronState_pos[CPU_thread_index];
	float * offset_JacAuxNeuronState_neg = JacAuxNeuronState_neg[CPU_thread_index];

	memcpy(offset_JacAuxNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
	Model->EvaluateDifferentialEcuation(offset_JacAuxNeuronState, offset_JacAuxNeuronState_pos);

	for (int j=0; j<N_DifferentialNeuronState; j++){
		memcpy(offset_JacAuxNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);

		offset_JacAuxNeuronState[j]-=epsi;
		Model->EvaluateDifferentialEcuation(offset_JacAuxNeuronState, offset_JacAuxNeuronState_neg);

		for(int z=0; z<N_DifferentialNeuronState; z++){
			jacnum[z*N_DifferentialNeuronState+j]=(offset_JacAuxNeuronState_pos[z]-offset_JacAuxNeuronState_neg[z])*inv_epsi;
		}
	} 
}





//With float (efficient 2)
void IntegrationMethod::invermat(float *a, float *ainv, int CPU_thread_index) {
	if(N_DifferentialNeuronState==1){
		ainv[0]=1.0f/a[0];
	}else{
		float coef, element, inv_element;
		int i,j, s;

		float * local_a= new float [N_DifferentialNeuronState*N_DifferentialNeuronState];
		float * local_ainv= new float [N_DifferentialNeuronState*N_DifferentialNeuronState]();

		memcpy(local_a, a, sizeof(float)*N_DifferentialNeuronState*N_DifferentialNeuronState);
		for (i=0;i<N_DifferentialNeuronState;i++){
			local_ainv[i*N_DifferentialNeuronState+i]=1.0f;
		}

		//Iteraciones
		for (s=0;s<N_DifferentialNeuronState;s++)
		{
			element=local_a[s*N_DifferentialNeuronState+s];

			if(element==0){
				for(int n=s+1; n<N_DifferentialNeuronState; n++){
					element=local_a[n*N_DifferentialNeuronState+s];
					if(element!=0){
						for(int m=0; m<N_DifferentialNeuronState; m++){
							float value=local_a[n*N_DifferentialNeuronState+m];
							local_a[n*N_DifferentialNeuronState+m]=local_a[s*N_DifferentialNeuronState+m];
							local_a[s*N_DifferentialNeuronState+m]=value;

							value=local_ainv[n*N_DifferentialNeuronState+m];
							local_ainv[n*N_DifferentialNeuronState+m]=local_ainv[s*N_DifferentialNeuronState+m];
							local_ainv[s*N_DifferentialNeuronState+m]=value;
						}
						break;
					}
					if(n==(N_DifferentialNeuronState-1)){
						printf("This matrix is not invertible\n");
						exit(0);
					}
				
				}
			}

			inv_element=1.0f/element;
			for (j=0;j<N_DifferentialNeuronState;j++){
				local_a[s*N_DifferentialNeuronState+j]*=inv_element;
				local_ainv[s*N_DifferentialNeuronState+j]*=inv_element;
			}

			for(i=0;i<N_DifferentialNeuronState;i++)
			{
				if (i!=s){
					coef=-local_a[i*N_DifferentialNeuronState+s];
					for (j=0;j<N_DifferentialNeuronState;j++){
						local_a[i*N_DifferentialNeuronState+j]+=local_a[s*N_DifferentialNeuronState+j]*coef;
						local_ainv[i*N_DifferentialNeuronState+j]+=local_ainv[s*N_DifferentialNeuronState+j]*coef;
					}
				}
			}
		}
		memcpy(ainv, local_ainv, sizeof(float)*N_DifferentialNeuronState*N_DifferentialNeuronState);
		delete [] local_a;
		delete [] local_ainv;
	}
}



//With float (with SSE3 intruction)
//void IntegrationMethod::invermat(float *a, float *ainv, int CPU_thread_index) {
//	if(N_DifferentialNeuronState==1){
//		ainv[0]=1.0f/a[0];
//	}else{
//		invermat_parallel(a, ainv, N_DifferentialNeuronState);
//	}
//}




