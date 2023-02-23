/***************************************************************************
 *                           BifixedStep.h                                 *
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

#ifndef BIFIXEDSTEP_H_
#define BIFIXEDSTEP_H_

/*!
 * \file BifixedStep.h
 *
 * \author Francisco Naveros
 * \date June 2015
 *
 * This file declares a class which abstracts all bi-fixed step integration methods in CPU.
 */

#include "./IntegrationMethodFast.h"


//class TimeDrivenNeuronModel;

/*!
 * \class BifixedStep
 *
 * \brief Bi-Fixed step integration methods
 *
 * This class abstracts the behavior of all bi-fixed step integration methods in CPU
 * for time-driven neural model.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date June 2015
 */
/*!
* Neuron model template
*/
template <class Neuron_Model>
class BifixedStep : public IntegrationMethodFast<Neuron_Model> {
	protected:

		/*!
 		 * \brief Default constructor of the class.
		 *
		 * It generates a new object.
		 */
		BifixedStep():IntegrationMethodFast<Neuron_Model>(),integrationMethodCounter(0),integrationMethodState(0){};

		/*!
		 * \brief Ratio between the large and the small integration steps (how many times the large step is larger than the small step).
		*/
		int ratioLargerSmallerSteps;

		/*!
		 * \brief Elapsed time in neuron model scale of the adaptative zone.
		*/
		float BifixedElapsedTimeInNeuronModelScale;

		/*!
		 * \brief Elapsed time in second of the adaptative zone.
		*/
		float BifixedElapsedTimeInSeconds;


		/*!
		 * \brief When the membrane potential reaches this value, the multi-step integration methods change the integration
		 *  step from elapsedTimeInNeuronModelScale to BifixedElapsedTimeInNeuronModelScale.
		*/
		float startVoltageThreshold;

		/*!
		 * \brief When the membrane potential reaches this value, the multi-step integration methods change the integration
		 *  step from BifixedElapsedTimeInNeuronModelScale to elapsedTimeInNeuronModelScale after timeAfterEndVoltageThreshold in seconds.
		*/
		float endVoltageThreshold;

		/*!
		 * \brief Number of small integration steps executed after the membrane potential reaches the endVoltageThreashold.
		*/
		int N_postBifixedSteps;

		/*!
		 * \brief Auxiliar values used to know which integration method must be used.
		*/
		int * integrationMethodCounter;

		/*!
		 * \brief Auxiliar values used to know which integration method must be used.
		*/
		int * integrationMethodState;


		/*!
 		 * \brief Default constructor of the class.
		 *
		 * It generates a new object using a pointer to a template class representing the neuron model.
		 *
		 * \param NewModel pointer to a template class representing the neuron model.
		 */
		BifixedStep(Neuron_Model *  NewModel):IntegrationMethodFast<Neuron_Model>(NewModel),integrationMethodCounter(0),
			integrationMethodState(0){
		}

	public:

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~BifixedStep(){
			if (this->integrationMethodCounter!=0){
				delete [] integrationMethodCounter;
			}
			if (this->integrationMethodState!=0){
				delete [] integrationMethodState;
			}
		}

		/*!
		 * \brief It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * \param index for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 * \param NeuronState neuron state variables of one neuron.
		 * \return Retrun if the neuron spike
		 */
		virtual void NextDifferentialEquationValues(int index, float * NeuronState) = 0;

		/*!
		* \brief It calculate the new neural state variables for a defined elapsed_time and all the neurons.
		*
		* It calculate the new neural state variables for a defined elapsed_time and all the neurons.
		*
		* \return Retrun if the neuron spike
		*/
		virtual void NextDifferentialEquationValues() = 0;

		/*!
		* \brief It calculate the new neural state variables for a defined elapsed_time and all the neurons that require integration.
		*
		* It calculate the new neural state variables for a defined elapsed_time and all the neurons that requre integration.
		*
		* \param integration_required array that sets if a neuron must be integrated (for lethargic neuron models)
		* \return Retrun if the neuron spike
		*/
		virtual void NextDifferentialEquationValues(bool * integration_required, double CurrentTime) = 0;

		/*!
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;


		/*!
		 * \brief It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param N_neuron number of neurons in the neuron model.
		 * \param inicialization vector with initial values.
		 */
		virtual void InitializeStates(int N_neurons, float * inicialization) = 0;


		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param index indicate which neuron must be reseted.
		 */
		virtual void resetState(int index) = 0;


		/*!
		 * \brief It sets the required parameter in the adaptative integration methods (Bifixed_Euler, Bifixed_RK2, Bifixed_RK4, Bifixed_BDF1 and Bifixed_BDF2).
		 *
		 * It sets the required parameter in the adaptative integration methods (Bifixed_Euler, Bifixed_RK2, Bifixed_RK4, Bifixed_BDF1 and Bifixed_BDF2).
		 *
		 * \param startVoltageThreshold, when the membrane potential reaches this value, the multi-step integration methods change the integration
		 *  step from elapsedTimeInNeuronModelScale to BifixedElapsedTimeInNeuronModelScale.
		 * \param endVoltageThreshold, when the membrane potential reaches this value, the multi-step integration methods change the integration
		 *  step from BifixedElapsedTimeInNeuronModelScale to ElapsedTimeInNeuronModelScale after timeAfterEndVoltageThreshold in seconds.
		 * \param timeAfterEndVoltageThreshold, time in neuron model scale ("seconds" or "miliseconds") that the multi-step integration methods maintain
		 *  the BifixedElapsedTimeInNeuronModelScale
		 *  after the endVoltageThreshold
		 */
		void SetBifixedStepParameters(float startVoltageThreshold, float endVoltageThreshold, float timeAfterEndVoltageThreshold){
			this->startVoltageThreshold=startVoltageThreshold;
			this->endVoltageThreshold=endVoltageThreshold;
			this->N_postBifixedSteps=ceil(timeAfterEndVoltageThreshold/BifixedElapsedTimeInNeuronModelScale)+1;
		}


		/*!
		 * \brief It calculates the conductance exponential values for time driven neuron models.
		 *
		 * It calculates the conductance exponential values for time driven neuron models.
		 */
		virtual void Calculate_conductance_exp_values()=0;

		/*!
		 * \brief It returns the integration method parameters.
		 *
		 * It returns the integration method parameters.
		 *
		 * \returns A dictionary with the integration method parameters
		 */
		virtual std::map<std::string, boost::any> GetParameters() const{
			// Return a dictionary with the parameters
			std::map<std::string, boost::any> newMap = IntegrationMethodFast<Neuron_Model>::GetParameters();
			newMap["n_steps"] = boost::any(this->ratioLargerSmallerSteps);
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
				this->ratioLargerSmallerSteps = newParam;
				param_map.erase(it);
			}

			IntegrationMethodFast<Neuron_Model>::SetParameters(param_map);

			//Bifixed step parameter initialization
			this->BifixedElapsedTimeInNeuronModelScale = this->elapsedTimeInNeuronModelScale / ratioLargerSmallerSteps;
			this->BifixedElapsedTimeInSeconds = this->elapsedTimeInSeconds / ratioLargerSmallerSteps;
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
			std::map<std::string, boost::any> newMap = IntegrationMethodFast<Neuron_Model>::GetDefaultParameters();
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
		 * \return An object with the parameters of the integration method.
		 */
		static ModelDescription ParseIntegrationMethod(FILE * fh, long Currentline) noexcept(false){
			ModelDescription nmodel = IntegrationMethodFast<Neuron_Model>::ParseIntegrationMethod(fh, Currentline);

			int ratioLargerSmallerStep;
			skip_comments(fh, Currentline);
			if (fscanf(fh, "%d", &ratioLargerSmallerStep) == 1){
				if (ratioLargerSmallerStep <= 0){
					throw EDLUTException(TASK_BI_FIXED_STEP_LOAD, ERROR_BI_FIXED_STEP_GLOBAL_LOCAL_RATIO, REPAIR_BI_FIXED_STEP);
				}
			}
			else{
				throw EDLUTException(TASK_BI_FIXED_STEP_LOAD, ERROR_BI_FIXED_STEP_READ_GLOBAL_LOCAL_RATIO, REPAIR_BI_FIXED_STEP);
			}
			nmodel.param_map["n_steps"] = boost::any(ratioLargerSmallerStep);
			return nmodel;
		}

        /*!
         * \brief Comparison operator between integration methods.
         *
         * It compares two integration methods.
         *
         * \return True if the integration methods are of the same type and with the same parameters.
         */
        virtual bool compare(const IntegrationMethod * rhs) const{
			if (!IntegrationMethodFast<Neuron_Model>::compare(rhs)){
				return false;
			}
			const BifixedStep * e = dynamic_cast<const BifixedStep *> (rhs);
			if (e == 0) return false;
			return this->ratioLargerSmallerSteps == e->ratioLargerSmallerSteps &&
					this->BifixedElapsedTimeInSeconds==e->BifixedElapsedTimeInSeconds &&
					this->startVoltageThreshold==e->startVoltageThreshold &&
					this->endVoltageThreshold==e->endVoltageThreshold &&
					this->N_postBifixedSteps==e->N_postBifixedSteps;
		};
};

#endif /* BIFIXEDSTEP_H_ */
