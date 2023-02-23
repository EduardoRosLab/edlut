/***************************************************************************
 *                           Euler.h                                       *
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

#ifndef EULER_H_
#define EULER_H_

/*!
 * \file Euler.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implements the Euler integration method. This class implement a fixed step
 * integration method.
 */

#include "./FixedStep.h"


/*!
 * \class Euler
 *
 * \brief Euler integration methods in CPU
 *
 * This class abstracts the behavior of a Euler integration method for neurons in a
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2013
 */

 /*!
 * Neuron model template
 */
template <class Neuron_Model>
class Euler : public FixedStep<Neuron_Model> {
    private:

    /*!
     * \brief Default constructor of the class.
     *
     * It generates a new object.
     */
		Euler():FixedStep<Neuron_Model>(){
		};

	public:

		/*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new Euler object.
		 *
		 * \param NewModel time driven neuron model associated to this integration method.
		 */
		Euler(Neuron_Model * NewModel) : FixedStep<Neuron_Model>(NewModel){
		};

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~Euler(){
		};


		/*!
		 * \brief It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * \param index for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 * \param NeuronState neuron state variables of one neuron.
		 * \return Retrun if the neuron spike
		 */
		void NextDifferentialEquationValues(int index, float * NeuronState){

			float previous_V = NeuronState[0];

			float AuxNeuronState[MAX_VARIABLES];

			this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, this->elapsedTimeInNeuronModelScale);

			for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				NeuronState[j] += this->elapsedTimeInNeuronModelScale*AuxNeuronState[j];
			}

			this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);

			//Update the last spike time.
			this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->elapsedTimeInSeconds);

			this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->elapsedTimeInNeuronModelScale);

			//Acumulate the membrane potential in a variable
			this->IncrementValidIntegrationVariable(NeuronState[0]);
		};


		/*!
		* \brief It calculate the new neural state variables for a defined elapsed_time and all the neurons.
		*
		* It calculate the new neural state variables for a defined elapsed_time and all the neurons.
		*
		* \return Retrun if the neuron spike
		*/
		void NextDifferentialEquationValues(){
			float AuxNeuronState[MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
				float previous_V = NeuronState[0];

				this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, this->elapsedTimeInNeuronModelScale);

				for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
					NeuronState[j] += this->elapsedTimeInNeuronModelScale*AuxNeuronState[j];
				}

				this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);

				//Update the last spike time.
				this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->elapsedTimeInSeconds);

				this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->elapsedTimeInNeuronModelScale);

				//Acumulate the membrane potential in a variable
				this->IncrementValidIntegrationVariable(NeuronState[0]);
			}
		};


		/*!
		* \brief It calculate the new neural state variables for a defined elapsed_time and all the neurons that require integration.
		*
		* It calculate the new neural state variables for a defined elapsed_time and all the neurons that requre integration.
		*
		* \param integration_required array that sets if a neuron must be integrated (for lethargic neuron models)
		* \return Retrun if the neuron spike
		*/
		void NextDifferentialEquationValues(bool * integration_required, double CurrentTime){
			float AuxNeuronState[MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				if (integration_required[index] == true){
					float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
					float previous_V = NeuronState[0];

					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, this->elapsedTimeInNeuronModelScale);

					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						NeuronState[j] += this->elapsedTimeInNeuronModelScale*AuxNeuronState[j];
					}

					this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);

					//Update the last spike time.
					this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->elapsedTimeInSeconds);

					this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->elapsedTimeInNeuronModelScale);

					//Set last update time for the analytic resolution of the differential equations in lethargic models
					this->neuron_model->State->SetLastUpdateTime(index, CurrentTime);

					//Acumulate the membrane potential in a variable
					this->IncrementValidIntegrationVariable(NeuronState[0]);
				}
			}
		};


		/*!
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out){
			out << "\t\tIntegration Method Type: " << Euler::GetName() << endl;
			out << "\t\tIntegration Step Time: " << this->elapsedTimeInSeconds<<"s" << endl;

			return out;
		};


		/*!
		 * \brief It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param N_neuron number of neurons in the neuron model.
		 * \param inicialization vector with initial values.
		 */
		void InitializeStates(int N_neurons, float * initialization){};

		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param index indicate which neuron must be reseted.
		 */
		void resetState(int index){};


		/*!
		 * \brief It calculates the conductance exponential values for time driven neuron models.
		 *
		 * It calculates the conductance exponential values for time driven neuron models.
		 */
		 void Calculate_conductance_exp_values(){
			 this->neuron_model->Initialize_conductance_exp_values(this->neuron_model->N_TimeDependentNeuronState, 1);
			 //index 0
			 this->neuron_model->Calculate_conductance_exp_values(0, this->elapsedTimeInNeuronModelScale);
		 };

		/*!
	   * \brief It returns the integration method parameters.
	   *
	   * It returns the integration method parameters.
	   *
	   * \returns A dictionary with the integration method parameters
	   */
		virtual std::map<std::string,boost::any> GetParameters() const{
			// Return a dictionary with the parameters
			std::map<std::string,boost::any> newMap = FixedStep<Neuron_Model>::GetParameters();
			newMap["name"] = Euler::GetName();
			return newMap;
		};

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
			FixedStep<Neuron_Model>::SetParameters(param_map);
		};

		/*!
		 * \brief It returns the default parameters of the integration method.
		 *
		 * It returns the default parameters of the integration method. It may be used to obtained the parameters that can be
		 * set for this integration method.
		 *
		 * \returns A dictionary with the integration method parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters(){
			std::map<std::string,boost::any> newMap = FixedStep<Neuron_Model>::GetDefaultParameters();
			newMap["name"] = Euler::GetName();
			return newMap;
		};

		/*!
		 * \brief It loads the integration method description.
		 *
		 * It loads the integration method description.
		 *
		 * \param fh Filehandler of the file with the information about the integration method.
		 *
		 * \return An object with the parameters of the integration method.
		 */
		static ModelDescription ParseIntegrationMethod(FILE * fh, long & Currentline) noexcept(false){
			ModelDescription nmodel = FixedStep<Neuron_Model>::ParseIntegrationMethod(fh, Currentline);
		    nmodel.model_name = Euler::GetName();
			return nmodel;
		};

		/*!
		 * \brief It returns the name of the integration method
		 *
		 * It returns the name of the integration method
		 */
		static std::string GetName(){
			return "Euler";
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
		static IntegrationMethod* CreateIntegrationMethod(ModelDescription nmDescription, Neuron_Model *nmodel){
		    Euler * newmodel = new Euler(nmodel);
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
        virtual bool compare(const IntegrationMethod * rhs) const{
            if (!FixedStep<Neuron_Model>::compare(rhs)){
                return false;
            }
            const Euler * e = dynamic_cast<const Euler *> (rhs);
            if (e == 0) return false;
            return true;
        };
};

#endif /* EULER_H_ */
