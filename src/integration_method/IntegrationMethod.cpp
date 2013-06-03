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


IntegrationMethod::IntegrationMethod(string integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState,int N_timeDependentNeuronState, int N_CPU_thread, bool jacobian, bool inverse):IntegrationMethodType(integrationMethodType), N_NeuronStateVariables(N_neuronStateVariables), N_DifferentialNeuronState(N_differentialNeuronState), N_TimeDependentNeuronState(N_timeDependentNeuronState){
	if(jacobian){
		AuxNeuronState = new float [N_NeuronStateVariables*N_CPU_thread]();
		AuxNeuronState_pos = new float [N_NeuronStateVariables*N_CPU_thread]();
		AuxNeuronState_neg = new float [N_NeuronStateVariables*N_CPU_thread]();
	}
	if(inverse){
		aux=new float[2*N_differentialNeuronState*N_CPU_thread];
		auxDouble=new double[2*N_differentialNeuronState*N_CPU_thread];
	}
}

IntegrationMethod::~IntegrationMethod(){
	free(AuxNeuronState);
	free(AuxNeuronState_pos);
	free(AuxNeuronState_neg);

	free(aux);
	free(auxDouble);
	free(PredictedElapsedTime);
}

string IntegrationMethod::GetType(){
	return this->IntegrationMethodType;
}
		
void IntegrationMethod::Jacobian(TimeDrivenNeuronModel * Model, float * NeuronState, float * jacnum, int CPU_thread_index){
	float epsi=9.5367431640625e-7;
	float * offset_AuxNeuronState = AuxNeuronState+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState_pos = AuxNeuronState_pos+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState_neg = AuxNeuronState_neg+(N_NeuronStateVariables*CPU_thread_index);
	for (int j=0; j<N_DifferentialNeuronState; j++){
		memcpy(offset_AuxNeuronState, NeuronState, sizeof(float)*N_NeuronStateVariables);
		offset_AuxNeuronState[j]+=epsi;
		Model->EvaluateDifferentialEcuation(offset_AuxNeuronState, offset_AuxNeuronState_pos);

		offset_AuxNeuronState[j]-=2*epsi;
		Model->EvaluateDifferentialEcuation(offset_AuxNeuronState, offset_AuxNeuronState_neg);

		for(int z=0; z<N_DifferentialNeuronState; z++){
			jacnum[z*N_DifferentialNeuronState+j]=(offset_AuxNeuronState_pos[z]-offset_AuxNeuronState_neg[z])/(2*epsi);
		}
	} 
}





//With float
void IntegrationMethod::invermat(float *a, float *ainv, int CPU_thread_index) {
	if(N_DifferentialNeuronState==1){
		ainv[0]=1/a[0];
	}else{
		float coef, element;
		float * offset_aux=aux+(2*N_DifferentialNeuronState*CPU_thread_index);
		int i,j, s;

		for (i=0;i<N_DifferentialNeuronState;i++){
			for(j=0;j<N_DifferentialNeuronState;j++){
				if(i==j)
					ainv[i*N_DifferentialNeuronState+j]=1.0;
				else
					ainv[i*N_DifferentialNeuronState+j]=0.0;
			}
		}

		//Iteraciones
		for (s=0;s<N_DifferentialNeuronState;s++)
		{
			element=a[s*N_DifferentialNeuronState+s];
			if(element==0){
				printf("Invertion matrix error\n");
				exit(0);
			}
			for (j=0;j<N_DifferentialNeuronState;j++){
				a[s*N_DifferentialNeuronState+j]/=element;
				ainv[s*N_DifferentialNeuronState+j]/=element;
			}

			for(i=0;i<N_DifferentialNeuronState;i++)
			{
				if (i!=s){
					coef=a[i*N_DifferentialNeuronState+s];
					for (j=0;j<N_DifferentialNeuronState;j++){
						offset_aux[j]=a[s*N_DifferentialNeuronState+j]*(coef*-1);
						offset_aux[N_DifferentialNeuronState+j]=ainv[s*N_DifferentialNeuronState+j]*(coef*-1);

					}
					for (j=0;j<N_DifferentialNeuronState;j++){
						a[i*N_DifferentialNeuronState+j]+=offset_aux[j];
						ainv[i*N_DifferentialNeuronState+j]+=offset_aux[N_DifferentialNeuronState+j];
					}
				}
			}
		}
	}
}

//With double
//void IntegrationMethod::invermat(float *a, float *ainv, int CPU_thread_index) {
//	if(N_DifferentialNeuronState==1){
//		ainv[0]=1/a[0];
//	}else{
//		double coef, element;
//		double * offset_auxDouble=auxDouble+(2*N_DifferentialNeuronState*CPU_thread_index);
//		int i,j, s;
//
//		double * aDouble =new double [N_DifferentialNeuronState*N_DifferentialNeuronState];
//		double * ainvDouble =new double [N_DifferentialNeuronState*N_DifferentialNeuronState];
//		
//		for (i=0;i<N_DifferentialNeuronState;i++){
//			for(j=0;j<N_DifferentialNeuronState;j++){
//				aDouble[i*N_DifferentialNeuronState+j]=(double)a[i*N_DifferentialNeuronState+j];
//				if(i==j)
//					ainvDouble[i*N_DifferentialNeuronState+j]=(double)1.0;
//				else
//					ainvDouble[i*N_DifferentialNeuronState+j]=(double)0.0;
//			}
//		}
//
//		//Iteraciones
//		for (s=0;s<N_DifferentialNeuronState;s++)
//		{
//			element=aDouble[s*N_DifferentialNeuronState+s];
//			if(element==0){
//				printf("Invertion matrix error\n");
//				exit(0);
//			}
//			for (j=0;j<N_DifferentialNeuronState;j++){
//				aDouble[s*N_DifferentialNeuronState+j]/=element;
//				ainvDouble[s*N_DifferentialNeuronState+j]/=element;
//			}
//
//			for(i=0;i<N_DifferentialNeuronState;i++)
//			{
//				if (i!=s){
//					coef=aDouble[i*N_DifferentialNeuronState+s];
//					for (j=0;j<N_DifferentialNeuronState;j++){
//						offset_auxDouble[j]=aDouble[s*N_DifferentialNeuronState+j]*(coef*-1);
//						offset_auxDouble[N_DifferentialNeuronState+j]=ainvDouble[s*N_DifferentialNeuronState+j]*(coef*-1);
//
//					}
//					for (j=0;j<N_DifferentialNeuronState;j++){
//						aDouble[i*N_DifferentialNeuronState+j]+=offset_auxDouble[j];
//						ainvDouble[i*N_DifferentialNeuronState+j]+=offset_auxDouble[N_DifferentialNeuronState+j];
//					}
//				}
//			}
//		}
//		for (i=0;i<N_DifferentialNeuronState;i++){
//			for(j=0;j<N_DifferentialNeuronState;j++){
//				ainv[i*N_DifferentialNeuronState+j]=(float)ainvDouble[i*N_DifferentialNeuronState+j];
//			}
//		}
//		free(aDouble);
//		free(ainvDouble);
//	}
//}

////With float and row change
//void IntegrationMethod::invermat(float *a, float *ainv, int CPU_thread_index) {
//	if(N_DifferentialNeuronState==1){
//		if(a[0]!=0){
//			ainv[0]=1/a[0];
//		}else{
//			printf("this matrix is not invertible");
//			exit(0);
//		}
//	}else{
//		float coef, element;
//		float * offset_aux=aux+(2*N_DifferentialNeuronState*CPU_thread_index);
//		int i,j, s;
//
//		for (i=0;i<N_DifferentialNeuronState;i++){
//			for(j=0;j<N_DifferentialNeuronState;j++){
//				if(i==j)
//					ainv[i*N_DifferentialNeuronState+j]=1.0;
//				else
//					ainv[i*N_DifferentialNeuronState+j]=0.0;
//			}
//		}
//
//		//Iteraciones
//		for (s=0;s<N_DifferentialNeuronState;s++)
//		{
//			element=a[s*N_DifferentialNeuronState+s];
//
//			if(element==0){
//				for(int n=s+1; n<N_DifferentialNeuronState; n++){
//					element=a[n*N_DifferentialNeuronState+s];
//					if(element!=0){
//						for(int m=0; m<N_DifferentialNeuronState; m++){
//							float value=a[n*N_DifferentialNeuronState+m];
//							a[n*N_DifferentialNeuronState+m]=a[s*N_DifferentialNeuronState+m];
//							a[s*N_DifferentialNeuronState+m]=value;
//						}
//						break;
//					}
//					if(n==(N_DifferentialNeuronState-1)){
//						printf("This matrix is not invertible\n");
//						exit(0);
//					}
//				
//				}
//			}
//
//			for (j=0;j<N_DifferentialNeuronState;j++){
//				a[s*N_DifferentialNeuronState+j]/=element;
//				ainv[s*N_DifferentialNeuronState+j]/=element;
//			}
//
//			for(i=0;i<N_DifferentialNeuronState;i++)
//			{
//				if (i!=s){
//					coef=a[i*N_DifferentialNeuronState+s];
//					for (j=0;j<N_DifferentialNeuronState;j++){
//						offset_aux[j]=a[s*N_DifferentialNeuronState+j]*(coef*-1);
//						offset_aux[N_DifferentialNeuronState+j]=ainv[s*N_DifferentialNeuronState+j]*(coef*-1);
//
//					}
//					for (j=0;j<N_DifferentialNeuronState;j++){
//						a[i*N_DifferentialNeuronState+j]+=offset_aux[j];
//						ainv[i*N_DifferentialNeuronState+j]+=offset_aux[N_DifferentialNeuronState+j];
//					}
//				}
//			}
//		}
//	}
//}

////With double and row change
//void IntegrationMethod::invermat(float *a, float *ainv, int CPU_thread_index) {
//	if(N_DifferentialNeuronState==1){
//		ainv[0]=1/a[0];
//	}else{
//		double coef, element;
//		double * offset_auxDouble=auxDouble+(2*N_DifferentialNeuronState*CPU_thread_index);
//		int i,j, s;
//
//		double * aDouble =new double [N_DifferentialNeuronState*N_DifferentialNeuronState];
//		double * ainvDouble =new double [N_DifferentialNeuronState*N_DifferentialNeuronState];
//		for (i=0;i<N_DifferentialNeuronState;i++){
//			for(j=0;j<N_DifferentialNeuronState;j++){
//				aDouble[i*N_DifferentialNeuronState+j]=(double)a[i*N_DifferentialNeuronState+j];
//				if(i==j)
//					ainvDouble[i*N_DifferentialNeuronState+j]=(double)1.0;
//				else
//					ainvDouble[i*N_DifferentialNeuronState+j]=(double)0.0;
//			}
//		}
//
//		//Iteraciones
//		for (s=0;s<N_DifferentialNeuronState;s++)
//		{
//			element=aDouble[s*N_DifferentialNeuronState+s];
//			if(element==0){
//				for(int n=s+1; n<N_DifferentialNeuronState; n++){
//					element=aDouble[n*N_DifferentialNeuronState+s];
//					if(element!=0){
//						for(int m=0; m<N_DifferentialNeuronState; m++){
//							double value=aDouble[n*N_DifferentialNeuronState+m];
//							aDouble[n*N_DifferentialNeuronState+m]=aDouble[s*N_DifferentialNeuronState+m];
//							aDouble[s*N_DifferentialNeuronState+m]=value;
//						}
//						break;
//					}
//					if(n==(N_DifferentialNeuronState-1)){
//						printf("This matrix is not invertible\n");
//						exit(0);
//					}
//				
//				}
//			}
//			for (j=0;j<N_DifferentialNeuronState;j++){
//				aDouble[s*N_DifferentialNeuronState+j]/=element;
//				ainvDouble[s*N_DifferentialNeuronState+j]/=element;
//			}
//
//			for(i=0;i<N_DifferentialNeuronState;i++)
//			{
//				if (i!=s){
//					coef=aDouble[i*N_DifferentialNeuronState+s];
//					for (j=0;j<N_DifferentialNeuronState;j++){
//						offset_auxDouble[j]=aDouble[s*N_DifferentialNeuronState+j]*(coef*-1);
//						offset_auxDouble[N_DifferentialNeuronState+j]=ainvDouble[s*N_DifferentialNeuronState+j]*(coef*-1);
//
//					}
//					for (j=0;j<N_DifferentialNeuronState;j++){
//						aDouble[i*N_DifferentialNeuronState+j]+=offset_auxDouble[j];
//						ainvDouble[i*N_DifferentialNeuronState+j]+=offset_auxDouble[N_DifferentialNeuronState+j];
//					}
//				}
//			}
//		}
//		for (i=0;i<N_DifferentialNeuronState;i++){
//			for(j=0;j<N_DifferentialNeuronState;j++){
//				ainv[i*N_DifferentialNeuronState+j]=(float)ainvDouble[i*N_DifferentialNeuronState+j];
//			}
//		}
//		free(aDouble);
//		free(ainvDouble);
//	}
//}