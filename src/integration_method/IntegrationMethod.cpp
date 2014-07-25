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






IntegrationMethod::IntegrationMethod(TimeDrivenNeuronModel* NewModel, string integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState,int N_timeDependentNeuronState, bool jacobian, bool inverse):model(NewModel),IntegrationMethodType(integrationMethodType), N_NeuronStateVariables(N_neuronStateVariables), N_DifferentialNeuronState(N_differentialNeuronState), N_TimeDependentNeuronState(N_timeDependentNeuronState){
	if(N_NeuronStateVariables>MAX_VARIABLES){
		cerr<<"The number of state variables is too high. You must increase the value of MAX_VARIABLES defined in IntegrationMethod.h to "<<N_NeuronStateVariables<<"."<<endl;
		exit(0);
	}
}

IntegrationMethod::~IntegrationMethod(){
	//delete [] PredictedElapsedTime;
}

string IntegrationMethod::GetType(){
	return this->IntegrationMethodType;
}
		


void IntegrationMethod::Jacobian(float * NeuronState, float * jacnum, float elapsed_time){
	float epsi=elapsed_time * 0.1f;
	float inv_epsi=1.0f/epsi;
	float JacAuxNeuronState[MAX_VARIABLES];
	float JacAuxNeuronState_pos[MAX_VARIABLES];
	float JacAuxNeuronState_neg[MAX_VARIABLES];

	memcpy(JacAuxNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
	this->model->EvaluateDifferentialEcuation(JacAuxNeuronState, JacAuxNeuronState_pos);

	for (int j=0; j<N_DifferentialNeuronState; j++){
		memcpy(JacAuxNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);

		JacAuxNeuronState[j]-=epsi;
		this->model->EvaluateDifferentialEcuation(JacAuxNeuronState, JacAuxNeuronState_neg);

		for(int z=0; z<N_DifferentialNeuronState; z++){
			jacnum[z*N_DifferentialNeuronState+j]=(JacAuxNeuronState_pos[z]-JacAuxNeuronState_neg[z])*inv_epsi;
		}
	} 
}





//With float (efficient 2)
void IntegrationMethod::invermat(float *a, float *ainv) {
	if(N_DifferentialNeuronState==1){
		ainv[0]=1.0f/a[0];
	}else{
		float coef, element, inv_element;
		int i,j, s;

		float local_a[MAX_VARIABLES*MAX_VARIABLES];
		float local_ainv[MAX_VARIABLES*MAX_VARIABLES]={};

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
	}
}







