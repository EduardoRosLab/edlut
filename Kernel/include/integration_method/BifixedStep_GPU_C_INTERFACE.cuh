/***************************************************************************
 *                           BifixedStep_GPU_C_INTERFACE.cuh               *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#ifndef BIFIXEDSTEP_GPU_C_INTERFACE_H_
#define BIFIXEDSTEP_GPU_C_INTERFACE_H_

/*!
 * \file BifixedStep_GPU_C_INTERFACE.h
 *
 * \author Francisco Naveros
 * \date May 2015
 *
 * This file declares a class which abstracts all multi step integration methods in GPU (this class is stored
 * in CPU memory and controles the allocation and deleting of GPU auxiliar memory).
 */


#include "integration_method/IntegrationMethodFast_GPU_C_INTERFACE.cuh"

#include "../../include/cudaError.h"
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/*!
 * \class BifixedStep_GPU_C_INTERFACE
 *
 * \brief Bifixed step integration method in CPU for GPU.
 *
 * This class abstracts the initializacion in CPU of multi step integration methods for GPU. This CPU class
 * controles the allocation and deleting of GPU auxiliar memory.
 *
 * \author Francisco Naveros
 * \date May 2015
 */
template <class Neuron_Model_GPU>
class BifixedStep_GPU_C_INTERFACE : public IntegrationMethodFast_GPU_C_INTERFACE<Neuron_Model_GPU>  {
	public:

		/*!
		 * \brief Number of multi step in the adapatative zone.
		*/
		int N_BifixedSteps;

		/*!
		 * \brief Elapsed time in neuron model scale of the adaptative zone.
		*/
		float BifixedElapsedTimeInNeuronModelScale;

		/*!
		 * \brief Elapsed time in second of the adaptative zone.
		*/
		float BifixedElapsedTimeInSeconds;

		/*!
		 * \brief Default constructor of the class.
		 *
		 * It generates a new object.
		 */
		BifixedStep_GPU_C_INTERFACE():IntegrationMethodFast_GPU_C_INTERFACE<Neuron_Model_GPU>(){};


		/*!
		* \brief Default constructor of the class.
		*
		* It generates a new object using a pointer to a template class representing the neuron model.
		*
		* \param NewModel pointer to a template class representing the neuron model.
		*/
		BifixedStep_GPU_C_INTERFACE(Neuron_Model_GPU *  NewModel) :IntegrationMethodFast_GPU_C_INTERFACE<Neuron_Model_GPU>(NewModel){
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~BifixedStep_GPU_C_INTERFACE(){
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
		 * \brief It returns the integration method parameters.
		 *
		 * It returns the integration method parameters.
		 *
		 * \returns A dictionary with the integration method parameters
		 */
		virtual std::map<std::string, boost::any> GetParameters() const{
			// Return a dictionary with the parameters
			std::map<std::string, boost::any> newMap = IntegrationMethodFast_GPU_C_INTERFACE<Neuron_Model_GPU>::GetParameters();
			newMap["n_steps"] = boost::any(this->N_BifixedSteps);
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
			std::map<std::string, boost::any>::iterator it = param_map.find("n_steps");
			if (it != param_map.end()){
				int newParam = boost::any_cast<int>(it->second);
				if (newParam <= 0){
					throw EDLUTException(TASK_FIXED_STEP_LOAD, ERROR_FIXED_STEP_STEP_SIZE, REPAIR_FIXED_STEP);
				}
				this->N_BifixedSteps = newParam;
				param_map.erase(it);
			}

			IntegrationMethodFast_GPU_C_INTERFACE<Neuron_Model_GPU>::SetParameters(param_map);

			//Bifixed step parameter initialization
			this->BifixedElapsedTimeInNeuronModelScale = this->elapsedTimeInNeuronModelScale / N_BifixedSteps;
			this->BifixedElapsedTimeInSeconds = this->elapsedTimeInSeconds / N_BifixedSteps;
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
			std::map<std::string, boost::any> newMap = IntegrationMethodFast_GPU_C_INTERFACE<Neuron_Model_GPU>::GetDefaultParameters();
			newMap["n_steps"] = boost::any(2);
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
			ModelDescription nmodel = IntegrationMethodFast_GPU_C_INTERFACE<Neuron_Model_GPU>::ParseIntegrationMethod(fh, Currentline);

			int N_BifixedStep;
			skip_comments(fh, Currentline);
			if (fscanf(fh, "%d", &N_BifixedStep) == 1){
				if (N_BifixedStep <= 0){
					throw EDLUTException(TASK_BI_FIXED_STEP_LOAD, ERROR_BI_FIXED_STEP_GLOBAL_LOCAL_RATIO, REPAIR_BI_FIXED_STEP);
				}
			}
			else{
				throw EDLUTException(TASK_BI_FIXED_STEP_LOAD, ERROR_BI_FIXED_STEP_READ_GLOBAL_LOCAL_RATIO, REPAIR_BI_FIXED_STEP);
			}
			nmodel.param_map["n_steps"] = boost::any(N_BifixedStep);
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
			if (!IntegrationMethodFast_GPU_C_INTERFACE<Neuron_Model_GPU>::compare(rhs)){
				return false;
			}
			const BifixedStep_GPU_C_INTERFACE *e = dynamic_cast<const BifixedStep_GPU_C_INTERFACE *> (rhs);
			if (e == 0) return false;
			return	this->N_BifixedSteps == e->N_BifixedSteps &&
				this->BifixedElapsedTimeInSeconds == e->BifixedElapsedTimeInSeconds;
        };
};

#endif /* BIFIXEDSTEP_GPU_C_INTERFACE_H_ */
