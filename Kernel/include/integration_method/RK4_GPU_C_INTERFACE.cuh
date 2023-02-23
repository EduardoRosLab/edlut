/***************************************************************************
 *                           RK4_GPU_C_INTERFACE.cuh                       *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef RK4_GPU_C_INTERFACE_H_
#define RK4_GPU_C_INTERFACE_H_

/*!
 * \file RK4_GPU_C_INTERFACE.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implements a fixed step fourth order Runge-Kutta integration method in GPU (this class is stored
 * in CPU memory and controles the allocation and deleting of GPU auxiliar memory).
 */

#include "../../include/integration_method/IntegrationMethod_GPU_GLOBAL_FUNCTIONS.cuh"
#include "integration_method/FixedStep_GPU_C_INTERFACE.cuh"

#include "../../include/cudaError.h"
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class RK4_GPU
 *
 * \brief RK4 integration method in CPU for GPU.
 *
 * This class abstracts the initializacion in CPU of a fourth order Runge-Kutta integration methods for GPU. This CPU class
 * controles the allocation and deleting of GPU auxiliar memory.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
template <class Neuron_Model_GPU>
class RK4_GPU_C_INTERFACE : public FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>{
	public:

		/*!
		 * \brief These vectors are used as auxiliar vectors.
		*/
		float * AuxNeuronState;
		float * AuxNeuronState1;
		float * AuxNeuronState2;
		float * AuxNeuronState3;
		float * AuxNeuronState4;

		/*!
     * \brief Default constructor of the class.
     *
     * It generates a new object.
     */
		RK4_GPU_C_INTERFACE():FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>(), AuxNeuronState(0), AuxNeuronState1(0), AuxNeuronState2(0), AuxNeuronState3(0), AuxNeuronState4(0){
		}


		/*!
		* \brief Default constructor of the class.
		*
		* It generates a new object using a pointer to a template class representing the neuron model.
		*
		* \param NewModel pointer to a template class representing the neuron model.
		*/

		RK4_GPU_C_INTERFACE(Neuron_Model_GPU *  NewModel) :FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>(NewModel), AuxNeuronState(0), AuxNeuronState1(0), AuxNeuronState2(0), AuxNeuronState3(0), AuxNeuronState4(0){
		}


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~RK4_GPU_C_INTERFACE(){
			if (AuxNeuronState != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState));
			}
			if (AuxNeuronState1 != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState1));
			}
			if (AuxNeuronState2 != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState2));
			}
			if (AuxNeuronState3 != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState3));
			}
			if (AuxNeuronState4 != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState4));
			}
		}


		/*!
		 * \brief This method reserves all the necesary GPU memory (this memory could be reserved directly in the GPU, but this
		 * suppose some restriction in the amount of memory which can be reserved).
		 *
		 * This method reserves all the necesary GPU memory (this memory could be reserved directly in the GPU, but this
		 * suppose some restriction in the amount of memory which can be reserved).
		 *
		 * \param N_neurons Number of neurons.
		 * \param Total_N_thread Number of thread in GPU.
		 */
		void InitializeMemoryGPU(int N_neurons, int Total_N_thread)
		{
			int size = 6 * sizeof(float *);

			cudaMalloc((void **)&this->Buffer_GPU, size);

			float integration_method_parameters_CPU[1];
			integration_method_parameters_CPU[0] = this->elapsedTimeInSeconds;
			float * integration_method_parameters_GPU;
			cudaMalloc((void**)&integration_method_parameters_GPU, 1 * sizeof(float));
			cudaMemcpy(integration_method_parameters_GPU, integration_method_parameters_CPU, 1 * sizeof(float), cudaMemcpyHostToDevice);

			cudaMalloc((void**)&AuxNeuronState, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&AuxNeuronState1, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&AuxNeuronState2, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&AuxNeuronState3, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&AuxNeuronState4, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));

			Call_RK4_GPU_C_INTERFACE_memory(this->Buffer_GPU, integration_method_parameters_GPU, AuxNeuronState, AuxNeuronState1, AuxNeuronState2, AuxNeuronState3, AuxNeuronState4);

			cudaFree(integration_method_parameters_GPU);
		}

		/*!
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		std::ostream & PrintInfo(std::ostream & out){
			out << "\t\tIntegration Method Type: " << RK4_GPU_C_INTERFACE::GetName() << endl;
			out << "\t\tIntegration Step Time: " << this->elapsedTimeInSeconds<<"s" << endl;

			return out;
		}

		/*!
		 * \brief It returns the integration method parameters.
		 *
		 * It returns the integration method parameters.
		 *
		 * \returns A dictionary with the integration method parameters
		 */
		virtual std::map<std::string, boost::any> GetParameters() const{
			std::map<std::string, boost::any> newMap = FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>::GetParameters();
			newMap["name"] = RK4_GPU_C_INTERFACE::GetName();
			return newMap;
		}

		/*!
		 * \brief It loads the integration method properties.
		 *
		 * It loads the integration method properties from parameter map.
		 *
		 * \param param_map The dictionary with the integration method parameters.
		 *
		 * \throw EDLUTFileException If it happens a mistake with the parameters in the dictionary.
		 */
		virtual void SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){
			// Search for the parameters in the dictionary
			FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>::SetParameters(param_map);
		}

		/*!
		 * \brief It returns the default parameters of the integration method.
		 *
		 * It returns the default parameters of the integration method. It may be used to obtained the parameters that can be
		 * set for this integration method.
		 *
		 * \returns A dictionary with the integration method parameters.
		 */
		static std::map<std::string, boost::any> GetDefaultParameters(){
			std::map<std::string, boost::any> newMap = FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>::GetDefaultParameters();
			newMap["name"] = RK4_GPU_C_INTERFACE::GetName();
			return newMap;
		}

        /*!
         * \brief It loads the integration method description.
         *
         * It loads the integration method description.
         *
         * \param fh Filehandler of the file with the information about the integration method.
         *
         * \return An object with the parameters of the neuron model.
         */
		static ModelDescription ParseIntegrationMethod(FILE * fh, long & Currentline) noexcept(false){
			ModelDescription nmodel = FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>::ParseIntegrationMethod(fh, Currentline);
			nmodel.model_name = RK4_GPU_C_INTERFACE::GetName();
			return nmodel;
		}

		/*!
		* \brief It returns the name of the integration method
		*
		* It returns the name of the integration method
		*/
		static std::string GetName(){
			return "RK4";
		};

		/*!
		* \brief It creates a new integration method object of this type.
		*
		* It creates a new integration method object of this type.
		*
		* \param param_map The integration method description object.
		*
		* \return A newly created integration method object.
		*/
		static IntegrationMethod_GPU_C_INTERFACE* CreateIntegrationMethod(ModelDescription nmDescription, Neuron_Model_GPU *nmodel){
			RK4_GPU_C_INTERFACE * newmodel = new RK4_GPU_C_INTERFACE(nmodel);
			newmodel->SetParameters(nmDescription.param_map);
			return newmodel;
		};

        /*!
         * \brief Comparison operator between integration methods.
         *
         * It compares two integration methods.
         *
         * \return True if the integration methods are of the same type and with the same parameters.
         */
        virtual bool compare(const IntegrationMethod_GPU_C_INTERFACE * rhs) const{
			if (!FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>::compare(rhs)){
				return false;
			}
			const RK4_GPU_C_INTERFACE *e = dynamic_cast<const RK4_GPU_C_INTERFACE *> (rhs);
			if (e == 0) return false;
			return true;
        };


};

#endif /* RK4_C_INTERFACE_GPU_H_ */
