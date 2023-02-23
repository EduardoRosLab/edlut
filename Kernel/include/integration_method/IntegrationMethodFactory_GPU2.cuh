/***************************************************************************
 *                           IntegrationMethodFactory_GPU2.cuh             *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Jesus Garrido                        *
 * email                : jesusgarrido@ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef INTEGRATIONMETHODFACTORY_GPU2_H
#define INTEGRATIONMETHODFACTORY_GPU2_H

#include <string>

#include "integration_method/Euler_GPU2.cuh"
#include "integration_method/RK2_GPU2.cuh"
#include "integration_method/RK4_GPU2.cuh"
#include "integration_method/BDFn_GPU2.cuh"
#include "integration_method/Bifixed_Euler_GPU2.cuh"
#include "integration_method/Bifixed_RK2_GPU2.cuh"
#include "integration_method/Bifixed_RK4_GPU2.cuh"
//#include "integration_method/Bifixed_BDFn_GPU2.cuh"

class IntegrationMethod_GPU2;


/*!
 * \file IntegrationMethodFactory_GPU2.cuh
 *
 * \author Jesus Garrido
 * \date June 2018
 *
 * This file declares a factory for IntegrationMethod object creation.
 */

/*!
* \class IntegrationMethodFactory
*
* \brief Integration method factory.
*
* This class declares the methods required for creating IntegrationMethod objects.
*
* \author Jesus Garrido
* \date June 2018
*/
template <class Neuron_Model>
class IntegrationMethodFactory_GPU2 {
	private:
		__device__ static int cmp4(char const* c1, char const* c2, int size){
			for (int j = 0; j<size; j++){
				if ((int)c1[j] >(int)c2[j]){
					return 1;
				}
				else if ((int)c1[j] < (int)c2[j]){
					return -1;
				}
			}
			return 0;
		}

//		__device__ static int atoiGPU(char const* data, int position){
//			return (((int)data[position]) - 48);
//		}

	public:

		__device__ static IntegrationMethod_GPU2 * loadIntegrationMethod_GPU2(char const* integrationName, void ** Buffer_GPU, Neuron_Model * neuronModel){
			IntegrationMethod_GPU2 * integrationMethod_GPU2;
			//DEFINE HERE NEW INTEGRATION METHOD
			if (IntegrationMethodFactory_GPU2::cmp4(integrationName, "Euler", 5) == 0){
				integrationMethod_GPU2 = (Euler_GPU2<Neuron_Model> *) new Euler_GPU2<Neuron_Model>(neuronModel, Buffer_GPU);
			}
			else if (IntegrationMethodFactory_GPU2::cmp4(integrationName, "RK2", 3) == 0){
				integrationMethod_GPU2 = (RK2_GPU2<Neuron_Model> *) new RK2_GPU2<Neuron_Model>(neuronModel, Buffer_GPU);
			}
			else if (IntegrationMethodFactory_GPU2::cmp4(integrationName, "RK4", 3) == 0){
				integrationMethod_GPU2 = (RK4_GPU2<Neuron_Model> *) new RK4_GPU2<Neuron_Model>(neuronModel, Buffer_GPU);
			}
			else if (IntegrationMethodFactory_GPU2::cmp4(integrationName, "BDF", 3) == 0){
				integrationMethod_GPU2 = (BDFn_GPU2<Neuron_Model> *) new BDFn_GPU2<Neuron_Model>(neuronModel, Buffer_GPU);
			}
			else if (IntegrationMethodFactory_GPU2::cmp4(integrationName, "Bifixed_Euler", 13) == 0){
				integrationMethod_GPU2 = (Bifixed_Euler_GPU2<Neuron_Model> *) new Bifixed_Euler_GPU2<Neuron_Model>(neuronModel, Buffer_GPU);
			}
			else if (IntegrationMethodFactory_GPU2::cmp4(integrationName, "Bifixed_RK2", 11) == 0){
				integrationMethod_GPU2 = (Bifixed_RK2_GPU2<Neuron_Model> *) new Bifixed_RK2_GPU2<Neuron_Model>(neuronModel, Buffer_GPU);
			}
			else if (IntegrationMethodFactory_GPU2::cmp4(integrationName, "Bifixed_RK4", 11) == 0){
				integrationMethod_GPU2 = (Bifixed_RK4_GPU2<Neuron_Model> *) new Bifixed_RK4_GPU2<Neuron_Model>(neuronModel, Buffer_GPU);
			}
			//else if (IntegrationMethodFactory_GPU2::cmp4(integrationName, "Bifixed_BDF", 11) == 0){
			//	integrationMethod_GPU2 = (Bifixed_BDFn_GPU2<Neuron_Model> *) new Bifixed_BDFn_GPU2<Neuron_Model>(neuronModel, Buffer_GPU);
			//}
			else{
				printf("There was an error loading the integration methods of the GPU.\n");
			}
			return integrationMethod_GPU2;
		}
	

};

#endif //EDLUT_INTEGRATIONMETHODFACTORY_GPU2_H
