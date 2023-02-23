/***************************************************************************
 *                           IntegrationMethodFast.h                       *
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

#ifndef INTEGRATIONMETHODFAST_H_
#define INTEGRATIONMETHODFAST_H_

/*!
 * \file IntegrationMethodFast.h
 *
 * \author Francisco Naveros
 * \date April 2019
 *
 * This file declares a class which abstracts all integration methods in CPU. This methods can
 * be fixed-step or bi-fixed-step. Finally, this class include a template reference to the
 * TimeDrivenNeuronModel derived class that must be integrated
 */

#include "./IntegrationMethod.h"
#include "./IntegrationMethodFactory.h"

#include "../neuron_model/TimeDrivenNeuronModel.h"
#include "../neuron_model/VectorNeuronState.h"

//class TimeDrivenNeuronModel;
//struct ModelDescription;



/*!
 * \class IntegrationMethod
 *
 * \brief Integration methods in CPU
 *
 * This class abstracts the behavior of all integration methods in CPU for time-driven neural model.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date May 2013
 */

/*!
* Neuron model template
*/
template <class Neuron_Model>
class IntegrationMethodFast : public IntegrationMethod {

	protected:

		/*
		* Time driven neuron model
		*/
		Neuron_Model * neuron_model;

	public:

		/*!
		* \brief Default Constructor without parameters.
		*
		* It generates a new IntegrationMethod object.
		*
		*/
		IntegrationMethodFast():IntegrationMethod(), neuron_model(0){
		}


		/*!
		* \brief Default Constructor without parameters.
		*
		* It generates a new IntegrationMethod object.
		*
		*/
		IntegrationMethodFast(Neuron_Model * NewModel):IntegrationMethod(), neuron_model(NewModel){
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~IntegrationMethodFast(){
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
		virtual std::ostream & PrintInfo(std::ostream & out) = 0;


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
		*  step from elapsedTimeInNeuronModelScale to bifixedElapsedTimeInNeuronModelScale.
		* \param endVoltageThreshold, when the membrane potential reaches this value, the multi-step integration methods change the integration
		*  step from bifixedElapsedTimeInNeuronModelScale to ElapsedTimeInNeuronModelScale after timeAfterEndVoltageThreshold in seconds.
		* \param timeAfterEndVoltageThreshold, time in neuron model scale ("seconds" or "miliseconds") that the multi-step integration methods maintain
		*  the bifixedElapsedTimeInNeuronModelScale
		*  after the endVoltageThreshold
		*/
		virtual void SetBifixedStepParameters(float startVoltageThreshold, float endVoltageThreshold, float timeAfterEndVoltageThreshold) = 0;

		/*!
		 * \brief It calculates the conductance exponential values for time driven neuron models.
		 *
		 * It calculates the conductance exponential values for time driven neuron models.
		 */
		virtual void Calculate_conductance_exp_values()=0;

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
			std::map<std::string, boost::any> newMap = IntegrationMethod::GetParameters();
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
			IntegrationMethod::SetParameters(param_map);
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
			std::map<std::string, boost::any> newMap = IntegrationMethod::GetDefaultParameters();
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
			ModelDescription nmodel = IntegrationMethod::ParseIntegrationMethod(fh, Currentline);
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
			if (!IntegrationMethod::compare(rhs)){
				return false;
			}
			const IntegrationMethodFast *e = dynamic_cast<const IntegrationMethodFast *> (rhs);
			if (e == 0) return false;
			return true;
        };
};

#endif /* INTEGRATIONMETHOD_H_ */
