/***************************************************************************
 *                           RK4.h                                         *
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

#ifndef RK4_H_
#define RK4_H_

/*!
 * \file RK4.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implements the fourth order Runge Kutta integration method. This class implement a fixed step
 * integration method.
 */

#include "./FixedStep.h"


/*!
 * \class RK4
 *
 * \brief RK4 integration methods in CPU
 *
 * This class abstracts the behavior of a fourth order Runge Kutta integration method for neurons in a
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
class RK4 : public FixedStep<Neuron_Model> {
	private:

		/*!
		  * \brief Default constructor of the class.
		 *
		 * It generates a new object.
		 */
		RK4():FixedStep<Neuron_Model>(){
		};

	public:


		/*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new fourth order Runge-Kutta object.
		 *
		 * \param NewModel time driven neuron model associated to this integration method.
		 */
		RK4(Neuron_Model * NewModel) : FixedStep<Neuron_Model>(NewModel){
		};

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~RK4(){
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

			int j;

			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState2[MAX_VARIABLES];
			float AuxNeuronState3[MAX_VARIABLES];
			float AuxNeuronState4[MAX_VARIABLES];


			const float elapsedTimeInNeuronModelScale_0_5 = this->elapsedTimeInNeuronModelScale*0.5f;
			const float elapsedTimeInNeuronModelScale_0_16 = this->elapsedTimeInNeuronModelScale*0.166666666667f;


			//1st term
			this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->elapsedTimeInNeuronModelScale);

			//2nd term
			for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				AuxNeuronState[j] = NeuronState[j] + AuxNeuronState1[j] * elapsedTimeInNeuronModelScale_0_5;
			}

			for (j = this->neuron_model->N_DifferentialNeuronState; j<this->neuron_model->N_NeuronStateVariables; j++){
				AuxNeuronState[j] = NeuronState[j];
			}

			this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState, index, 0);
			this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState, AuxNeuronState2, index, this->elapsedTimeInNeuronModelScale);

			//3rd term
			for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				AuxNeuronState[j] = NeuronState[j] + AuxNeuronState2[j] * elapsedTimeInNeuronModelScale_0_5;
			}

			this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState, AuxNeuronState3, index, this->elapsedTimeInNeuronModelScale);

			//4rd term
			for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				AuxNeuronState[j] = NeuronState[j] + AuxNeuronState3[j] * this->elapsedTimeInNeuronModelScale;
			}

			this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState, index, 0);
			this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState, AuxNeuronState4, index, this->elapsedTimeInNeuronModelScale);


			for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				NeuronState[j] += (AuxNeuronState1[j] + 2.0f*(AuxNeuronState2[j] + AuxNeuronState3[j]) + AuxNeuronState4[j])*elapsedTimeInNeuronModelScale_0_16;
			}

			//Finaly, we evaluate the neural state variables with time dependence.
			for (j = this->neuron_model->N_DifferentialNeuronState; j<this->neuron_model->N_NeuronStateVariables; j++){
				NeuronState[j] = AuxNeuronState[j];
			}

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

			int j;

			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState2[MAX_VARIABLES];
			float AuxNeuronState3[MAX_VARIABLES];
			float AuxNeuronState4[MAX_VARIABLES];

			const float elapsedTimeInNeuronModelScale_0_5 = this->elapsedTimeInNeuronModelScale*0.5f;
			const float elapsedTimeInNeuronModelScale_0_16 = this->elapsedTimeInNeuronModelScale*0.166666666667f;

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
				float previous_V = NeuronState[0];
				//1st term
				this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->elapsedTimeInNeuronModelScale);

				//2nd term
				for (j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
					AuxNeuronState[j] = NeuronState[j] + AuxNeuronState1[j] * elapsedTimeInNeuronModelScale_0_5;
				}

				for (j = this->neuron_model->N_DifferentialNeuronState; j < this->neuron_model->N_NeuronStateVariables; j++){
					AuxNeuronState[j] = NeuronState[j];
				}

				this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState, index, 0);
				this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState, AuxNeuronState2, index, this->elapsedTimeInNeuronModelScale);

				//3rd term
				for (j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
					AuxNeuronState[j] = NeuronState[j] + AuxNeuronState2[j] * elapsedTimeInNeuronModelScale_0_5;
				}

				this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState, AuxNeuronState3, index, this->elapsedTimeInNeuronModelScale);

				//4rd term
				for (j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
					AuxNeuronState[j] = NeuronState[j] + AuxNeuronState3[j] * this->elapsedTimeInNeuronModelScale;
				}

				this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState, index, 0);
				this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState, AuxNeuronState4, index, this->elapsedTimeInNeuronModelScale);


				for (j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
					NeuronState[j] += (AuxNeuronState1[j] + 2.0f*(AuxNeuronState2[j] + AuxNeuronState3[j]) + AuxNeuronState4[j])*elapsedTimeInNeuronModelScale_0_16;
				}

				//Finaly, we evaluate the neural state variables with time dependence.
				for (j = this->neuron_model->N_DifferentialNeuronState; j < this->neuron_model->N_NeuronStateVariables; j++){
					NeuronState[j] = AuxNeuronState[j];
				}

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

			int j;

			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState2[MAX_VARIABLES];
			float AuxNeuronState3[MAX_VARIABLES];
			float AuxNeuronState4[MAX_VARIABLES];

			const float elapsedTimeInNeuronModelScale_0_5 = this->elapsedTimeInNeuronModelScale*0.5f;
			const float elapsedTimeInNeuronModelScale_0_16 = this->elapsedTimeInNeuronModelScale*0.166666666667f;

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				if (integration_required[index] == true){
					float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
					float previous_V = NeuronState[0];
					//1st term
					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->elapsedTimeInNeuronModelScale);

					//2nd term
					for (j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState[j] = NeuronState[j] + AuxNeuronState1[j] * elapsedTimeInNeuronModelScale_0_5;
					}

					for (j = this->neuron_model->N_DifferentialNeuronState; j < this->neuron_model->N_NeuronStateVariables; j++){
						AuxNeuronState[j] = NeuronState[j];
					}

					this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState, index, 0);
					this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState, AuxNeuronState2, index, this->elapsedTimeInNeuronModelScale);

					//3rd term
					for (j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState[j] = NeuronState[j] + AuxNeuronState2[j] * elapsedTimeInNeuronModelScale_0_5;
					}

					this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState, AuxNeuronState3, index, this->elapsedTimeInNeuronModelScale);

					//4rd term
					for (j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState[j] = NeuronState[j] + AuxNeuronState3[j] * this->elapsedTimeInNeuronModelScale;
					}

					this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState, index, 0);
					this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState, AuxNeuronState4, index, this->elapsedTimeInNeuronModelScale);


					for (j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						NeuronState[j] += (AuxNeuronState1[j] + 2.0f*(AuxNeuronState2[j] + AuxNeuronState3[j]) + AuxNeuronState4[j])*elapsedTimeInNeuronModelScale_0_16;
					}

					//Finaly, we evaluate the neural state variables with time dependence.
					for (j = this->neuron_model->N_DifferentialNeuronState; j < this->neuron_model->N_NeuronStateVariables; j++){
						NeuronState[j] = AuxNeuronState[j];
					}

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
			out << "\t\tIntegration Method Type: " << RK4::GetName() << endl;
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
			this->neuron_model->Calculate_conductance_exp_values(0, this->elapsedTimeInNeuronModelScale*0.5f);
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
			newMap["name"] = RK4::GetName();
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
			newMap["name"] = RK4::GetName();
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
		    nmodel.model_name = RK4::GetName();
			return nmodel;
		};

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
		static IntegrationMethod* CreateIntegrationMethod(ModelDescription nmDescription, Neuron_Model *nmodel){
		    RK4 * newmodel = new RK4(nmodel);
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
            const RK4 * e = dynamic_cast<const RK4 *> (rhs);
            if (e == 0) return false;
            return true;
        };
};

#endif /* RK4_H_ */
