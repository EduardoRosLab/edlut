/***************************************************************************
 *                           IntegrationMethodFast_GPU_C_INTERFACE.cuh     *
 *                           -------------------                           *
 * copyright            : (C) 2019 by Francisco Naveros                    *
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

#ifndef INTEGRATIONMETHODFAST_GPU_C_INTERFACE_H_
#define INTEGRATIONMETHODFAST_GPU_C_INTERFACE_H_

/*!
 * \file IntegrationMethodFast_GPU_C_INTERFACE.cuh
 *
 * \author Francisco Naveros
 * \date April 2019
 *
 * This file declares a class which abstracts all integration methods in GPU. This methods can
 * be fixed-step or bi-fixed-step. Finally, this class include a template reference to the 
 * TimeDrivenNeuronModel_GPU_C_INTERFACE derived class that must be integrated
 */

#include "./IntegrationMethod_GPU_C_INTERFACE.cuh"
#include "./IntegrationMethodFactory_GPU_C_INTERFACE.cuh"

#include "neuron_model/TimeDrivenNeuronModel_GPU_C_INTERFACE.cuh"
#include "neuron_model/VectorNeuronState_GPU_C_INTERFACE.cuh"

#include "../../include/cudaError.h"
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//class TimeDrivenNeuronModel;
//struct ModelDescription;



/*!
 * \class IntegrationMethodFast_GPU_C_INTERFACE
 *
 * \brief Integration methods in CPU for GPU
 *
 * This class abstracts the behavior of all integration methods in GPU for time-driven neural model.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date May 2019
 */

/*!
* Neuron model template
*/
template <class Neuron_Model_GPU>
class IntegrationMethodFast_GPU_C_INTERFACE : public IntegrationMethod_GPU_C_INTERFACE {

	protected:
		
		/*
		* Time driven neuron model
		*/
		Neuron_Model_GPU * neuron_model;

	public:

		/*!
		* \brief Default Constructor without parameters.
		*
		* It generates a new IntegrationMethod object.
		*
		*/
		IntegrationMethodFast_GPU_C_INTERFACE(Neuron_Model_GPU * NewModel) : IntegrationMethod_GPU_C_INTERFACE(), neuron_model(NewModel){
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~IntegrationMethodFast_GPU_C_INTERFACE(){
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
		virtual void InitializeMemoryGPU(int N_neurons, int Total_N_thread)=0;


		/*!
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual std::ostream & PrintInfo(std::ostream & out) = 0;


		/*!
		* \brief It sets the integration time step of this integration method in neuron model scale.
		*
		* It sets the integration time step of this integration method in neuron model scale.
		*/
		void SetIntegrationTimeStepNeuronModel(){
			this->elapsedTimeInNeuronModelScale = this->elapsedTimeInSeconds*this->neuron_model->GetTimeScale();
		}

		/*!
		 * \brief It returns the integration method parameters.
		 *
		 * It returns the integration method parameters.
		 *
		 * \returns A dictionary with the integration method parameters
		 */
		virtual std::map<std::string, boost::any> GetParameters() const{
			std::map<std::string, boost::any> newMap = IntegrationMethod_GPU_C_INTERFACE::GetParameters();
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
			IntegrationMethod_GPU_C_INTERFACE::SetParameters(param_map);
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
			std::map<std::string, boost::any> newMap = IntegrationMethod_GPU_C_INTERFACE::GetDefaultParameters();
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
			ModelDescription nmodel = IntegrationMethod_GPU_C_INTERFACE::ParseIntegrationMethod(fh, Currentline);
			return nmodel;
		}

        /*!
         * \brief Comparison operator between integration methods.
         *
         * It compares two integration methods.
         *
         * \return True if the integration methods are of the same type and with the same parameters.
         */
        virtual bool compare(const IntegrationMethod_GPU_C_INTERFACE * rhs) const{
			if (!IntegrationMethod_GPU_C_INTERFACE::compare(rhs)){
				return false;
			}
			const IntegrationMethodFast_GPU_C_INTERFACE *e = dynamic_cast<const IntegrationMethodFast_GPU_C_INTERFACE *> (rhs);
			if (e == 0) return false;
			return true;
        };
};

#endif /* INTEGRATIONMETHODFAST_GPU_C_INTERFACE_H_ */
