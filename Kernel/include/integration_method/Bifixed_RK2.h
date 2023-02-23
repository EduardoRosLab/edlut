/***************************************************************************
 *                           Bifixed_RK2.h                                 *
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

#ifndef Bifixed_RK2_H_
#define Bifixed_RK2_H_

/*!
 * \file Bifixed_RK2.h
 *
 * \author Francisco Naveros
 * \date May 2015
 *
 * This file declares a class which implements an adaptive second order Runge Kutta integration method. This class implement a multi step
 * integration method.
 */

#include "./BifixedStep.h"


/*!
 * \class Bifixed_RK2
 *
 * \brief Bifixed_RK2 integration methods in CPU
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
class Bifixed_RK2 : public BifixedStep<Neuron_Model> {

    private:

        /*!
          * \brief Default constructor of the class.
         *
         * It generates a new object.
         */
		Bifixed_RK2():BifixedStep<Neuron_Model>(){
		};

    public:

		/*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new second order Runge-Kutta object.
		 *
		 * \param NewModel time driven neuron model associated to this integration method.
		 */
		Bifixed_RK2(Neuron_Model * NewModel) : BifixedStep<Neuron_Model>(NewModel){
		};

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~Bifixed_RK2(){
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
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState2[MAX_VARIABLES];
			float PreviousNeuronState[MAX_VARIABLES];

			if (this->integrationMethodState[index] == 0){
				memcpy(PreviousNeuronState, NeuronState, sizeof(float)*this->neuron_model->N_NeuronStateVariables);

				//1st term
				this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->elapsedTimeInNeuronModelScale);

				//2nd term
				for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					NeuronState[j] += AuxNeuronState1[j] * this->elapsedTimeInNeuronModelScale;
				}

				this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);
				this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState2, index, this->elapsedTimeInNeuronModelScale);


				for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					NeuronState[j] += (-AuxNeuronState1[j] + AuxNeuronState2[j])*this->elapsedTimeInNeuronModelScale*0.5f;
				}

				if (NeuronState[0]>this->startVoltageThreshold){
					this->integrationMethodState[index] = 1;
					//Restore the neuron model state to a previous state.
					memcpy(NeuronState, PreviousNeuronState, sizeof(float)*this->neuron_model->N_NeuronStateVariables);
				}
				else{
					//Update the last spike time.
					this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->elapsedTimeInSeconds);

					//Acumulate the membrane potential in a variable
					this->IncrementValidIntegrationVariable(NeuronState[0]);
				}
			}

			if (this->integrationMethodState[index]>0){
				for (int iteration = 0; iteration<this->ratioLargerSmallerSteps; iteration++){
					float previous_V = NeuronState[0];

					//1st term
					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->BifixedElapsedTimeInNeuronModelScale);

					//2nd term
					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						NeuronState[j] += AuxNeuronState1[j] * this->BifixedElapsedTimeInNeuronModelScale;
					}


					this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 1);
					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState2, index, this->BifixedElapsedTimeInNeuronModelScale);


					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						NeuronState[j] += (-AuxNeuronState1[j] + AuxNeuronState2[j])*this->BifixedElapsedTimeInNeuronModelScale*0.5f;
					}


					//Update the last spike time.
					this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->BifixedElapsedTimeInSeconds);

					this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);

					//Acumulate the membrane potential in a variable
					this->IncrementValidIntegrationVariable(NeuronState[0]);

					if (NeuronState[0]>this->startVoltageThreshold && this->integrationMethodState[index] == 1){
						this->integrationMethodState[index] = 2;
					}
					else if (NeuronState[0]<this->endVoltageThreshold && this->integrationMethodState[index] == 2){
						this->integrationMethodState[index] = 3;
						this->integrationMethodCounter[index] = this->N_postBifixedSteps;
					}
					if (this->integrationMethodCounter[index]>0 && this->integrationMethodState[index] == 3){
						this->integrationMethodCounter[index]--;
						if (this->integrationMethodCounter[index] == 0){
							this->integrationMethodState[index] = 0;
						}
					}
				}
				if (this->integrationMethodState[index] == 1){
					this->integrationMethodState[index] = 0;
				}
			}
		}


		/*!
		* \brief It calculate the new neural state variables for a defined elapsed_time and all the neurons.
		*
		* It calculate the new neural state variables for a defined elapsed_time and all the neurons.
		*
		* \return Retrun if the neuron spike
		*/
		void NextDifferentialEquationValues(){
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState2[MAX_VARIABLES];
			float PreviousNeuronState[MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);

				if (this->integrationMethodState[index] == 0){
					memcpy(PreviousNeuronState, NeuronState, sizeof(float)*this->neuron_model->N_NeuronStateVariables);

					//1st term
					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->elapsedTimeInNeuronModelScale);

					//2nd term
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						NeuronState[j] += AuxNeuronState1[j] * this->elapsedTimeInNeuronModelScale;
					}

					this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);
					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState2, index, this->elapsedTimeInNeuronModelScale);


					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						NeuronState[j] += (-AuxNeuronState1[j] + AuxNeuronState2[j])*this->elapsedTimeInNeuronModelScale*0.5f;
					}

					if (NeuronState[0]>this->startVoltageThreshold){
						this->integrationMethodState[index] = 1;
						//Restore the neuron model state to a previous state.
						memcpy(NeuronState, PreviousNeuronState, sizeof(float)*this->neuron_model->N_NeuronStateVariables);
					}
					else{
						//Update the last spike time.
						this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->elapsedTimeInSeconds);

						//Acumulate the membrane potential in a variable
						this->IncrementValidIntegrationVariable(NeuronState[0]);
					}
				}

				if (this->integrationMethodState[index] > 0){
					for (int iteration = 0; iteration < this->ratioLargerSmallerSteps; iteration++){
						float previous_V = NeuronState[0];

						//1st term
						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->BifixedElapsedTimeInNeuronModelScale);

						//2nd term
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							NeuronState[j] += AuxNeuronState1[j] * this->BifixedElapsedTimeInNeuronModelScale;
						}


						this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 1);
						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState2, index, this->BifixedElapsedTimeInNeuronModelScale);


						for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
							NeuronState[j] += (-AuxNeuronState1[j] + AuxNeuronState2[j])*this->BifixedElapsedTimeInNeuronModelScale*0.5f;
						}


						//Update the last spike time.
						this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->BifixedElapsedTimeInSeconds);

						this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);

						//Acumulate the membrane potential in a variable
						this->IncrementValidIntegrationVariable(NeuronState[0]);

						if (NeuronState[0]>this->startVoltageThreshold && this->integrationMethodState[index] == 1){
							this->integrationMethodState[index] = 2;
						}
						else if (NeuronState[0] < this->endVoltageThreshold && this->integrationMethodState[index] == 2){
							this->integrationMethodState[index] = 3;
							this->integrationMethodCounter[index] = this->N_postBifixedSteps;
						}
						if (this->integrationMethodCounter[index]>0 && this->integrationMethodState[index] == 3){
							this->integrationMethodCounter[index]--;
							if (this->integrationMethodCounter[index] == 0){
								this->integrationMethodState[index] = 0;
							}
						}

					}
					if (this->integrationMethodState[index] == 1){
						this->integrationMethodState[index] = 0;
					}
				}
			}
		}



		/*!
		* \brief It calculate the new neural state variables for a defined elapsed_time and all the neurons that require integration.
		*
		* It calculate the new neural state variables for a defined elapsed_time and all the neurons that requre integration.
		*
		* \param integration_required array that sets if a neuron must be integrated (for lethargic neuron models)
		* \return Retrun if the neuron spike
		*/
		void NextDifferentialEquationValues(bool * integration_required, double CurrentTime){
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState2[MAX_VARIABLES];
			float PreviousNeuronState[MAX_VARIABLES];


			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				if (integration_required[index] == true){
					float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);

					if (this->integrationMethodState[index] == 0){
						memcpy(PreviousNeuronState, NeuronState, sizeof(float)*this->neuron_model->N_NeuronStateVariables);

						//1st term
						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->elapsedTimeInNeuronModelScale);

						//2nd term
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							NeuronState[j] += AuxNeuronState1[j] * this->elapsedTimeInNeuronModelScale;
						}

						this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);
						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState2, index, this->elapsedTimeInNeuronModelScale);


						for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
							NeuronState[j] += (-AuxNeuronState1[j] + AuxNeuronState2[j])*this->elapsedTimeInNeuronModelScale*0.5f;
						}

						if (NeuronState[0]>this->startVoltageThreshold){
							this->integrationMethodState[index] = 1;
							//Restore the neuron model state to a previous state.
							memcpy(NeuronState, PreviousNeuronState, sizeof(float)*this->neuron_model->N_NeuronStateVariables);
						}
						else{
							//Update the last spike time.
							this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->elapsedTimeInSeconds);

							//Acumulate the membrane potential in a variable
							this->IncrementValidIntegrationVariable(NeuronState[0]);
						}
					}

					if (this->integrationMethodState[index] > 0){
						for (int iteration = 0; iteration < this->ratioLargerSmallerSteps; iteration++){
							float previous_V = NeuronState[0];

							//1st term
							this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->BifixedElapsedTimeInNeuronModelScale);

							//2nd term
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								NeuronState[j] += AuxNeuronState1[j] * this->BifixedElapsedTimeInNeuronModelScale;
							}


							this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 1);
							this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState2, index, this->BifixedElapsedTimeInNeuronModelScale);


							for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
								NeuronState[j] += (-AuxNeuronState1[j] + AuxNeuronState2[j])*this->BifixedElapsedTimeInNeuronModelScale*0.5f;
							}


							//Update the last spike time.
							this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->BifixedElapsedTimeInSeconds);

							this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);

							//Acumulate the membrane potential in a variable
							this->IncrementValidIntegrationVariable(NeuronState[0]);

							if (NeuronState[0]>this->startVoltageThreshold && this->integrationMethodState[index] == 1){
								this->integrationMethodState[index] = 2;
							}
							else if (NeuronState[0] < this->endVoltageThreshold && this->integrationMethodState[index] == 2){
								this->integrationMethodState[index] = 3;
								this->integrationMethodCounter[index] = this->N_postBifixedSteps;
							}
							if (this->integrationMethodCounter[index]>0 && this->integrationMethodState[index] == 3){
								this->integrationMethodCounter[index]--;
								if (this->integrationMethodCounter[index] == 0){
									this->integrationMethodState[index] = 0;
								}
							}
						}
						if (this->integrationMethodState[index] == 1){
							this->integrationMethodState[index] = 0;
						}
					}

					//Set last update time for the analytic resolution of the differential equations in lethargic models
					this->neuron_model->State->SetLastUpdateTime(index, CurrentTime);
				}
			}
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
		virtual ostream & PrintInfo(ostream & out){
			out << "\t\tIntegration Method Type: " << Bifixed_RK2::GetName() << endl;
			out << "\t\tIntegration Step Time: " << this->elapsedTimeInSeconds<<"s" << endl;
			out << "\t\tNumber of Bifixed Steps: " << this->ratioLargerSmallerSteps << endl;

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
		void InitializeStates(int N_neurons, float * initialization){
			this->integrationMethodCounter = new int[N_neurons]();
			this->integrationMethodState = new int[N_neurons]();
		}

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
			this->neuron_model->Initialize_conductance_exp_values(this->neuron_model->N_TimeDependentNeuronState, 2);
			//index 0
			this->neuron_model->Calculate_conductance_exp_values(0, this->elapsedTimeInNeuronModelScale);
			//index 1
			this->neuron_model->Calculate_conductance_exp_values(1, this->BifixedElapsedTimeInNeuronModelScale);
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
			std::map<std::string,boost::any> newMap = BifixedStep<Neuron_Model>::GetParameters();
			newMap["name"] = Bifixed_RK2::GetName();
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
			BifixedStep<Neuron_Model>::SetParameters(param_map);
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
			std::map<std::string,boost::any> newMap = BifixedStep<Neuron_Model>::GetDefaultParameters();
			newMap["name"] = Bifixed_RK2::GetName();
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
			ModelDescription nmodel = BifixedStep<Neuron_Model>::ParseIntegrationMethod(fh, Currentline);
		    nmodel.model_name = Bifixed_RK2::GetName();
			return nmodel;
		};

		/*!
		 * \brief It returns the name of the integration method
		 *
		 * It returns the name of the integration method
		 */
		static std::string GetName(){
			return "Bifixed_RK2";
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
		    Bifixed_RK2 * newmodel = new Bifixed_RK2(nmodel);
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
            if (!BifixedStep<Neuron_Model>::compare(rhs)){
                return false;
            }
            const Bifixed_RK2 * e = dynamic_cast<const Bifixed_RK2 *> (rhs);
            if (e == 0) return false;
            return true;
        };
};

#endif /* BIFIXED_RK2_H_ */
