/***************************************************************************
 *                           LoadIntegrationMethod_GPU.h                   *
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

#ifndef LOADINTEGRATIONMETHOD_GPU2_H_
#define LOADINTEGRATIONMETHOD_GPU2_H_

/*!
 * \file LoadIntegrationMethod_GPU.h
 *
 * \author Francisco Naveros
 * \date November 2012
 *
 * This file declares a class which load all integration methods in GPU.
 */

#include <string>
//using namespace std;


#include "./IntegrationMethod_GPU2.h"
#include "./Euler_GPU2.h"
#include "./RK2_GPU2.h"
#include "./RK4_GPU2.h"
#include "./BDFn_GPU2.h"





/*!
 * \class LoadIntegrationMethod_GPU2
 *
 * \brief Load Integration methods in GPU
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class LoadIntegrationMethod_GPU2 {

	public:

		__device__ static int cmp4(char const* c1, char const* c2, int size){
			for(int j=0; j<size; j++){
				if((int)c1[j] > (int)c2[j]){
					return 1;
				}else if((int)c1[j] < (int)c2[j]){
					return -1;
				}
			}
			return 0;
		}

		__device__ static int atoiGPU(char const* data, int position){
			return (((int)data[position])-48);
		}


		__device__ static IntegrationMethod_GPU2 * loadIntegrationMethod_GPU2(char const* integrationName, int N_NeuronStateVariables, int N_DifferentialNeuronState, int N_TimeDependentNeuronState, int Total_N_thread, void ** Buffer_GPU){
			
			IntegrationMethod_GPU2 * Method;
			//DEFINE HERE NEW INTEGRATION METHOD
			if(cmp4(integrationName, "Euler", 5)==0){
				Method=(Euler_GPU2 *) new Euler_GPU2(N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState, Total_N_thread, Buffer_GPU);
			}else if(cmp4(integrationName, "RK2", 3)==0){
				Method=(RK2_GPU2 *) new RK2_GPU2(N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState, Total_N_thread,Buffer_GPU);
			}else if(cmp4(integrationName, "RK4", 3)==0){
				Method=(RK4_GPU2 *) new RK4_GPU2(N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState, Total_N_thread,Buffer_GPU);
			}else if(cmp4(integrationName, "BDF", 3)==0 && atoiGPU(integrationName,3)>0 && atoiGPU(integrationName,3)<7){
				Method=(BDFn_GPU2 *) new BDFn_GPU2(N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState, Total_N_thread, Buffer_GPU, atoiGPU(integrationName,3));
			}else{
				printf("There was an error loading the integration methods of the GPU.\n");

			}
			return Method;
		}

		

};




#endif /* LOADINTEGRATIONMETHOD_GPU2_H_ */
