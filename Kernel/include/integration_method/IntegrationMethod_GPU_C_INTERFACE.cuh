/***************************************************************************
 *                           IntegrationMethod_GPU_C_INTERFACE.cuh         *
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

#ifndef INTEGRATIONMETHOD_GPU_C_INTERFACE_H_
#define INTEGRATIONMETHOD_GPU_C_INTERFACE_H_

/*!
 * \file IntegrationMethod_GPU_C_INTERFACE.cuh
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which abstracts all integration methods in GPU (this class is stored
 * in CPU memory and controles the allocation and deleting of GPU auxiliar memory). This methods can
 * be fixed-step or bi-fixed-step.
 */

#include "../../include/integration_method/IntegrationMethod_GPU2.cuh"

//#include <string>
//#include <string.h>
#include <map>
#include <boost/any.hpp>

#include "simulation/Utils.h"
//#include "spike/EDLUTFileException.h"
#include "spike/EDLUTException.h"
#include "simulation/NetworkDescription.h"

struct ModelDescription;

using namespace std;


#include "../../include/cudaError.h"
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



/*!
 * \class IntegrationMethod_GPU_C_INTERFACE
 *
 * \brief Integration method in CPU for GPU.
 *
 * This class abstracts the initializacion in CPU of integration methods for GPU. This CPU class
 * controles the allocation and deleting of GPU auxiliar memory.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class IntegrationMethod_GPU_C_INTERFACE {
	public:

		/*!
		 * \brief This vector contains all the necesary GPU memory which have been reserved in the CPU (this memory
		 * could be reserved directly in the GPU, but this suppose some restriction in the amount of memory which can be reserved).
		*/
		void ** Buffer_GPU;

		/*!
		 * \brief Integration step size in seconds (the time scale of the simulator).
		*/
		float elapsedTimeInSeconds;

		/*!
		 * \brief Integration step size in seconds or miliseconds, depending on the neuron model that is going to be integrated.
		*/
		float elapsedTimeInNeuronModelScale;

		/*!
		* \brief Integration method name.
		*/
		std::string name;


		/*!
		 * \brief Constructor of the class.
		 *
		 * It generates a new IntegrationMethod_GPU_C_INTERFACE object.
		 *
		 */

		IntegrationMethod_GPU_C_INTERFACE() :Buffer_GPU(0), elapsedTimeInSeconds(0), elapsedTimeInNeuronModelScale(0), name(""){
		}


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~IntegrationMethod_GPU_C_INTERFACE(){
			if (Buffer_GPU != 0){
				HANDLE_ERROR(cudaFree(Buffer_GPU));
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
		* \brief It gets the integration time step of this integration method.
		*
		* It gets the integration time step of this integration method.
		*
		* \return The integration time step (in seconds).
		*/
		float GetIntegrationTimeStep() const {
			return this->elapsedTimeInSeconds;
		}


		/*!
		* \brief It sets the integration time step of this integration method in neuron model scale.
		*
		* It sets the integration time step of this integration method in neuron model scale.
		*/
		virtual void SetIntegrationTimeStepNeuronModel() = 0;

		/*!
		* \brief It returns the integration method parameters.
		*
		* It returns the integration method parameters.
		*
		* \returns A dictionary with the integration method parameters
		*/
		virtual std::map<std::string, boost::any> GetParameters() const{
			// Return a dictionary with the parameters
			std::map<std::string, boost::any> newMap;
			newMap["step"] = boost::any(this->elapsedTimeInSeconds);
			newMap["name"] = boost::any(this->name);
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
			std::map<std::string, boost::any>::iterator it = param_map.find("step");
			if (it != param_map.end()){
				float newParam = boost::any_cast<float>(it->second);
				if (newParam <= 0.0){
					throw EDLUTException(TASK_FIXED_STEP_LOAD, ERROR_FIXED_STEP_STEP_SIZE, REPAIR_FIXED_STEP);
				}
				this->elapsedTimeInSeconds = newParam;
				SetIntegrationTimeStepNeuronModel();
				param_map.erase(it);
			}

			it = param_map.find("name");
			if (it != param_map.end()){
				std::string newParam = boost::any_cast<std::string>(it->second);
				this->name = newParam;
				param_map.erase(it);
			}

			if (!param_map.empty()){
				throw EDLUTException(TASK_INTEGRATION_METHOD_SET, ERROR_INTEGRATION_METHOD_UNKNOWN_PARAMETER, REPAIR_INTEGRATION_METHOD_PARAMETER_NAME);
			}
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
			std::map<std::string, boost::any> newMap;
			newMap["step"] = boost::any(0.001f);
			newMap["name"] = boost::any(std::string("IntegrationMethod"));
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
			ModelDescription nmodel;
			skip_comments(fh, Currentline);
			float elapsedTime;
			if (fscanf(fh, "%f", &elapsedTime) != 1) {
				throw EDLUTException(TASK_FIXED_STEP_LOAD, ERROR_FIXED_STEP_READ_STEP, REPAIR_FIXED_STEP);
			}
			nmodel.param_map["step"] = boost::any(elapsedTime);
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
			return this->elapsedTimeInSeconds == rhs->elapsedTimeInSeconds;
		};
};

#endif /* INTEGRATIONMETHOD_GPU_C_INTERFACE_H_ */
