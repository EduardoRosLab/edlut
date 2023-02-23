/***************************************************************************
 *                           Bifixed_BDFn.h                                *
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

#ifndef BIFIXED_BDFn_H_
#define BIFIXED_BDFn_H_

/*!
 * \file Bifixed_BDFn.h
 *
 * \author Francisco Naveros
 * \date May 2015
 *
 * This file declares a class which implements two BDF (Backward Differentiation Formulas) integration methods
 * (first and second order multi step BDF integration method). This method implements a progressive implementation of the
 * higher order integration method using the lower order integration mehtod (BDF1->BDF2). This class
 * implement a multi step integration method.
 */

#include "./BifixedStep.h"


/*!
 * \class Bifixed_BDFn
 *
 * \brief Bifixed_BDFn integration methods in CPU
 *
 * This class abstracts the behavior of BDF1 and BDF2 integration methods for neurons in a
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2015
 */
 /*!
 * Neuron model template
 */
template <class Neuron_Model>
class Bifixed_BDFn : public BifixedStep<Neuron_Model> {
    private:

        /*!
          * \brief Default constructor of the class.
         *
         * It generates a new BDF object.
         */
        Bifixed_BDFn():BifixedStep<Neuron_Model>(), BDForder(1), PreviousNeuronState(0), D(0), PreviousIntegrationStep(0), CoeficientSelector(0), state(0){
		};

    public:

		/*!
		 * \brief This vector stores previous neuron state variable for all neurons. This one is used as a memory.
		*/
		float ** PreviousNeuronState;

		/*!
		 * \brief This vector stores the difference between previous neuron state variable for all neurons. This
		 * one is used as a memory.
		*/
		float ** D;

		/*!
		 * \brief This constant matrix stores the coefficients of each BDF order. Additionally, the coeficiente
		 * that must be used depend on the previous and actual integration step.
		*/
		float Coeficient [3][3][4];//decrement of h, equal h, increment of h

		/*!
		 * \brief Integration step used in previous integration
		*/
		float * PreviousIntegrationStep;

		/*!
		 * \brief This variable indicate which BDF coeficients must be used in funcion of the previous and actual integration step.
		*/
		int * CoeficientSelector;

		/*!
		 * \brief This vector contains the state of each neuron (BDF order). When the integration method is reseted (the values of the neuron model variables are
		 * changed outside the integration method, for instance when a neuron spikes and the membrane potential is reseted to the resting potential), the values
		 * store in PreviousNeuronState and D are no longer valid. In this case the order it is set to 0 and must grow in each integration step until it is reache
		 * the target order.
		*/
		int * state;

		/*!
		 * \brief This value stores the order of the integration method.
		*/
		int BDForder;


		/*!
		 * \brief Constructor of the class with parameters.
		 *
		 * It generates a new BDF object indicating the order of the method.
		 *
		 * \param NewModel time driven neuron model associated to this integration method.
		 */
		Bifixed_BDFn(Neuron_Model * NewModel) : BifixedStep<Neuron_Model>(NewModel), BDForder(1), PreviousNeuronState(0), D(0), PreviousIntegrationStep(0), CoeficientSelector(0), state(0){
		};

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~Bifixed_BDFn(){
			if (BDForder > 1 && PreviousNeuronState != 0){
				for (int i = 0; i<BDForder - 1; i++){
					delete[] PreviousNeuronState[i];
				}
				delete[] PreviousNeuronState;
			}

			if (D != 0){
				for (int i = 0; i < BDForder; i++){
					delete[] D[i];
				}
				delete[] D;
			}

			if (state != 0){
				delete[] state;
			}

			if(PreviousIntegrationStep != 0){
				delete [] PreviousIntegrationStep;
			}

			if(CoeficientSelector != 0){
				delete [] CoeficientSelector;
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
		void NextDifferentialEquationValues(int index, float * NeuronState){

			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState_p[MAX_VARIABLES];
			float AuxNeuronState_p1[MAX_VARIABLES];
			float AuxNeuronState_c[MAX_VARIABLES];
			float jacnum[MAX_VARIABLES*MAX_VARIABLES];
			float J[MAX_VARIABLES*MAX_VARIABLES];
			float inv_J[MAX_VARIABLES*MAX_VARIABLES];

			if (this->integrationMethodState[index] == 0){
				//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
				if (state[index] == 0){
					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, this->elapsedTimeInNeuronModelScale);
					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_p[j] = NeuronState[j] + this->elapsedTimeInNeuronModelScale*AuxNeuronState[j];
					}
				}
				//In this case we use the value of previous states to calculate an aproximation of the solution.
				else{
					float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / this->elapsedTimeInNeuronModelScale;
					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_p[j] = NeuronState[j];
						for (int i = 0; i<state[index]; i++){
							AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
						}
					}
				}

				for (int i = this->neuron_model->N_DifferentialNeuronState; i<this->neuron_model->N_NeuronStateVariables; i++){
					AuxNeuronState_p[i] = NeuronState[i];
				}


				this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 0);



				float epsi = 1.0f;
				int k = 0;

				this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->elapsedTimeInNeuronModelScale);

				//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
				//epsi is the difference between two consecutive aproximation of the implicit method.
				while (epsi>1e-16 && k<5){
					this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, this->elapsedTimeInNeuronModelScale);

					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->elapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * this->elapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
						for (int i = 1; i<state[index]; i++){
							AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
					}

					//jacobian.
					Jacobian(AuxNeuronState_p, jacnum, index);

					for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
						for (int t = 0; t<this->neuron_model->N_DifferentialNeuronState; t++){
							J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->elapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
							if (z == t){
								J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
							}
						}
					}

					this->invermat(J, inv_J);

					for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
						float aux = 0.0;
						for (int t = 0; t<this->neuron_model->N_DifferentialNeuronState; t++){
							aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
						}
						AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
					}

					//We calculate the difference between both aproximations.
					float aux = 0.0f;
					float aux2 = 0.0f;
					for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
						aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
						if (aux>aux2){
							aux2 = aux;
						}
					}

					memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

					epsi = aux2;
					k++;
				}

				if (NeuronState[0]>this->startVoltageThreshold){
					this->integrationMethodState[index] = 1;

					//Restore the neuron model state to a previous state.
					this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, -this->elapsedTimeInSeconds);

					//Comes form a small step and goes to a small step
					if (CoeficientSelector[index] == 2){
						CoeficientSelector[index] = 1;
					}
					else{//goes to a smaller step.
						CoeficientSelector[index] = 0;
					}
				}
				else{
					//We increase the state of the integration method.
					if (state[index]<BDForder){
						state[index]++;
					}

					//We acumulate these new values for the next step.
					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){

						for (int i = (state[index] - 1); i>0; i--){
							D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
						D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
						for (int i = 1; i<state[index]; i++){
							D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
					}

					if (state[index]>1){
						for (int i = state[index] - 2; i>0; i--){
							memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
						}

						memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
					}
					memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



					//Finaly, we evaluate the neural state variables with time dependence.
					this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);

					//update the integration step size.
					PreviousIntegrationStep[index] = this->elapsedTimeInNeuronModelScale;

					//Set the coeficient selector to 1 for the next iteration.
					CoeficientSelector[index] = 1;

					//Update the last spike time.
					this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->elapsedTimeInSeconds);

					//Acumulate the membrane potential in a variable
					this->IncrementValidIntegrationVariable(NeuronState[0]);
				}
			}

			if (this->integrationMethodState[index]>0){

				for (int iteration = 0; iteration<this->ratioLargerSmallerSteps; iteration++){
					float previous_V = NeuronState[0];

					//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
					if (state[index] == 0){
						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);
						for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j] = NeuronState[j] + this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j];
						}
					}
					//In this case we use the value of previous states to calculate an aproximation of the solution.
					else{
						float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / this->BifixedElapsedTimeInNeuronModelScale;
						for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j] = NeuronState[j];
							for (int i = 0; i<state[index]; i++){
								AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
							}
						}
					}

					for (int i = this->neuron_model->N_DifferentialNeuronState; i<this->neuron_model->N_NeuronStateVariables; i++){
						AuxNeuronState_p[i] = NeuronState[i];
					}

					this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 1);

					float epsi = 1.0f;
					int k = 0;


					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->BifixedElapsedTimeInNeuronModelScale);

					//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
					//epsi is the difference between two consecutive aproximation of the implicit method.
					while (epsi>1e-16 && k<5){
						this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);

						for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
							for (int i = 1; i<state[index]; i++){
								AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
						}

						//jacobian.
						Jacobian(AuxNeuronState_p, jacnum, index);

						for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
							for (int t = 0; t<this->neuron_model->N_DifferentialNeuronState; t++){
								J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->BifixedElapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
								if (z == t){
									J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
								}
							}
						}

						this->invermat(J, inv_J);

						for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
							float aux = 0.0;
							for (int t = 0; t<this->neuron_model->N_DifferentialNeuronState; t++){
								aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
							}
							AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
						}

						//We calculate the difference between both aproximations.
						float aux = 0.0f;
						float aux2 = 0.0f;
						for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
							aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
							if (aux>aux2){
								aux2 = aux;
							}
						}

						memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

						epsi = aux2;
						k++;
					}

					//We increase the state of the integration method.
					if (state[index]<BDForder){
						state[index]++;
					}


					//We acumulate these new values for the next step.
					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){

						for (int i = (state[index] - 1); i>0; i--){
							D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
						D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
						for (int i = 1; i<state[index]; i++){
							D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
					}

					if (state[index]>1){
						for (int i = state[index] - 2; i>0; i--){
							memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
						}

						memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
					}
					memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



					//Finaly, we evaluate the neural state variables with time dependence.
					this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 1);

					//Update the last spike time.
					this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->BifixedElapsedTimeInSeconds);

					this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);

					//Acumulate the membrane potential in a variable
					this->IncrementValidIntegrationVariable(NeuronState[0]);

					//Set the CoeficientSelector to 1.
					CoeficientSelector[index] = 1;
					PreviousIntegrationStep[index] = this->BifixedElapsedTimeInNeuronModelScale;

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
							CoeficientSelector[index] = 2;
						}
					}
				}
				if (this->integrationMethodState[index] == 1){
					this->integrationMethodState[index] = 0;
					CoeficientSelector[index] = 2;
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

			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState_p[MAX_VARIABLES];
			float AuxNeuronState_p1[MAX_VARIABLES];
			float AuxNeuronState_c[MAX_VARIABLES];
			float jacnum[MAX_VARIABLES*MAX_VARIABLES];
			float J[MAX_VARIABLES*MAX_VARIABLES];
			float inv_J[MAX_VARIABLES*MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
				if (this->integrationMethodState[index] == 0){
					//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
					if (state[index] == 0){
						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, this->elapsedTimeInNeuronModelScale);
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j] = NeuronState[j] + this->elapsedTimeInNeuronModelScale*AuxNeuronState[j];
						}
					}
					//In this case we use the value of previous states to calculate an aproximation of the solution.
					else{
						float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / this->elapsedTimeInNeuronModelScale;
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j] = NeuronState[j];
							for (int i = 0; i < state[index]; i++){
								AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
							}
						}
					}

					for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
						AuxNeuronState_p[i] = NeuronState[i];
					}


					this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 0);



					float epsi = 1.0f;
					int k = 0;

					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->elapsedTimeInNeuronModelScale);

					//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
					//epsi is the difference between two consecutive aproximation of the implicit method.
					while (epsi > 1e-16 && k < 5){
						this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, this->elapsedTimeInNeuronModelScale);

						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->elapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * this->elapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
							for (int i = 1; i < state[index]; i++){
								AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
						}

						//jacobian.
						Jacobian(AuxNeuronState_p, jacnum, index);

						for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
							for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
								J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->elapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
								if (z == t){
									J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
								}
							}
						}

						this->invermat(J, inv_J);

						for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
							float aux = 0.0;
							for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
								aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
							}
							AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
						}

						//We calculate the difference between both aproximations.
						float aux = 0.0f;
						float aux2 = 0.0f;
						for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
							aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
							if (aux > aux2){
								aux2 = aux;
							}
						}

						memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

						epsi = aux2;
						k++;
					}

					if (NeuronState[0] > this->startVoltageThreshold){
						this->integrationMethodState[index] = 1;

						//Restore the neuron model state to a previous state.
						this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, -this->elapsedTimeInSeconds);

						//Comes form a small step and goes to a small step
						if (CoeficientSelector[index] == 2){
							CoeficientSelector[index] = 1;
						}
						else{//goes to a smaller step.
							CoeficientSelector[index] = 0;
						}
					}
					else{
						//We increase the state of the integration method.
						if (state[index] < BDForder){
							state[index]++;
						}

						//We acumulate these new values for the next step.
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){

							for (int i = (state[index] - 1); i > 0; i--){
								D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
							D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
							for (int i = 1; i < state[index]; i++){
								D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
						}

						if (state[index]>1){
							for (int i = state[index] - 2; i > 0; i--){
								memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
							}

							memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
						}
						memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



						//Finaly, we evaluate the neural state variables with time dependence.
						this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);

						//update the integration step size.
						PreviousIntegrationStep[index] = this->elapsedTimeInNeuronModelScale;

						//Set the coeficient selector to 1 for the next iteration.
						CoeficientSelector[index] = 1;

						//Update the last spike time.
						this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->elapsedTimeInSeconds);

						//Acumulate the membrane potential in a variable
						this->IncrementValidIntegrationVariable(NeuronState[0]);
					}
				}

				if (this->integrationMethodState[index] > 0){

					for (int iteration = 0; iteration < this->ratioLargerSmallerSteps; iteration++){
						float previous_V = NeuronState[0];

						//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
						if (state[index] == 0){
							this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_p[j] = NeuronState[j] + this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j];
							}
						}
						//In this case we use the value of previous states to calculate an aproximation of the solution.
						else{
							float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / this->BifixedElapsedTimeInNeuronModelScale;
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_p[j] = NeuronState[j];
								for (int i = 0; i < state[index]; i++){
									AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
								}
							}
						}

						for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
							AuxNeuronState_p[i] = NeuronState[i];
						}

						this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 1);

						float epsi = 1.0f;
						int k = 0;


						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->BifixedElapsedTimeInNeuronModelScale);

						//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
						//epsi is the difference between two consecutive aproximation of the implicit method.
						while (epsi > 1e-16 && k < 5){
							this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);

							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
								for (int i = 1; i < state[index]; i++){
									AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
							}

							//jacobian.
							Jacobian(AuxNeuronState_p, jacnum, index);

							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
									J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->BifixedElapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
									if (z == t){
										J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
									}
								}
							}

							this->invermat(J, inv_J);

							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								float aux = 0.0;
								for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
									aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
								}
								AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
							}

							//We calculate the difference between both aproximations.
							float aux = 0.0f;
							float aux2 = 0.0f;
							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
								if (aux > aux2){
									aux2 = aux;
								}
							}

							memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

							epsi = aux2;
							k++;
						}

						//We increase the state of the integration method.
						if (state[index] < BDForder){
							state[index]++;
						}


						//We acumulate these new values for the next step.
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){

							for (int i = (state[index] - 1); i > 0; i--){
								D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
							D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
							for (int i = 1; i < state[index]; i++){
								D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
						}

						if (state[index]>1){
							for (int i = state[index] - 2; i > 0; i--){
								memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
							}

							memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
						}
						memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



						//Finaly, we evaluate the neural state variables with time dependence.
						this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 1);

						//Update the last spike time.
						this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->BifixedElapsedTimeInSeconds);

						this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);

						//Acumulate the membrane potential in a variable
						this->IncrementValidIntegrationVariable(NeuronState[0]);

						//Set the CoeficientSelector to 1.
						CoeficientSelector[index] = 1;
						PreviousIntegrationStep[index] = this->BifixedElapsedTimeInNeuronModelScale;

						if (NeuronState[0] > this->startVoltageThreshold && this->integrationMethodState[index] == 1){
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
								CoeficientSelector[index] = 2;
							}
						}
					}
					if (this->integrationMethodState[index] == 1){
						this->integrationMethodState[index] = 0;
						CoeficientSelector[index] = 2;
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

			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState_p[MAX_VARIABLES];
			float AuxNeuronState_p1[MAX_VARIABLES];
			float AuxNeuronState_c[MAX_VARIABLES];
			float jacnum[MAX_VARIABLES*MAX_VARIABLES];
			float J[MAX_VARIABLES*MAX_VARIABLES];
			float inv_J[MAX_VARIABLES*MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				if (integration_required[index] == true){
					float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
					if (this->integrationMethodState[index] == 0){
						//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
						if (state[index] == 0){
							this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, this->elapsedTimeInNeuronModelScale);
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_p[j] = NeuronState[j] + this->elapsedTimeInNeuronModelScale*AuxNeuronState[j];
							}
						}
						//In this case we use the value of previous states to calculate an aproximation of the solution.
						else{
							float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / this->elapsedTimeInNeuronModelScale;
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_p[j] = NeuronState[j];
								for (int i = 0; i < state[index]; i++){
									AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
								}
							}
						}

						for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
							AuxNeuronState_p[i] = NeuronState[i];
						}


						this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 0);



						float epsi = 1.0f;
						int k = 0;

						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->elapsedTimeInNeuronModelScale);

						//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
						//epsi is the difference between two consecutive aproximation of the implicit method.
						while (epsi > 1e-16 && k < 5){
							this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, this->elapsedTimeInNeuronModelScale);

							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->elapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * this->elapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
								for (int i = 1; i < state[index]; i++){
									AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
							}

							//jacobian.
							Jacobian(AuxNeuronState_p, jacnum, index);

							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
									J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->elapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
									if (z == t){
										J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
									}
								}
							}

							this->invermat(J, inv_J);

							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								float aux = 0.0;
								for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
									aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
								}
								AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
							}

							//We calculate the difference between both aproximations.
							float aux = 0.0f;
							float aux2 = 0.0f;
							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
								if (aux > aux2){
									aux2 = aux;
								}
							}

							memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

							epsi = aux2;
							k++;
						}

						if (NeuronState[0] > this->startVoltageThreshold){
							this->integrationMethodState[index] = 1;

							//Restore the neuron model state to a previous state.
							this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, -this->elapsedTimeInSeconds);

							//Comes form a small step and goes to a small step
							if (CoeficientSelector[index] == 2){
								CoeficientSelector[index] = 1;
							}
							else{//goes to a smaller step.
								CoeficientSelector[index] = 0;
							}
						}
						else{
							//We increase the state of the integration method.
							if (state[index] < BDForder){
								state[index]++;
							}

							//We acumulate these new values for the next step.
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){

								for (int i = (state[index] - 1); i > 0; i--){
									D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
								D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
								for (int i = 1; i < state[index]; i++){
									D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
							}

							if (state[index]>1){
								for (int i = state[index] - 2; i > 0; i--){
									memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
								}

								memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
							}
							memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



							//Finaly, we evaluate the neural state variables with time dependence.
							this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);

							//update the integration step size.
							PreviousIntegrationStep[index] = this->elapsedTimeInNeuronModelScale;

							//Set the coeficient selector to 1 for the next iteration.
							CoeficientSelector[index] = 1;

							//Update the last spike time.
							this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->elapsedTimeInSeconds);

							//Acumulate the membrane potential in a variable
							this->IncrementValidIntegrationVariable(NeuronState[0]);
						}
					}

					if (this->integrationMethodState[index] > 0){

						for (int iteration = 0; iteration < this->ratioLargerSmallerSteps; iteration++){
							float previous_V = NeuronState[0];

							//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
							if (state[index] == 0){
								this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);
								for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
									AuxNeuronState_p[j] = NeuronState[j] + this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j];
								}
							}
							//In this case we use the value of previous states to calculate an aproximation of the solution.
							else{
								float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / this->BifixedElapsedTimeInNeuronModelScale;
								for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
									AuxNeuronState_p[j] = NeuronState[j];
									for (int i = 0; i < state[index]; i++){
										AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
									}
								}
							}

							for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
								AuxNeuronState_p[i] = NeuronState[i];
							}

							this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 1);

							float epsi = 1.0f;
							int k = 0;


							this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, this->BifixedElapsedTimeInNeuronModelScale);

							//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
							//epsi is the difference between two consecutive aproximation of the implicit method.
							while (epsi > 1e-16 && k < 5){
								this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);

								for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
									AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
									for (int i = 1; i < state[index]; i++){
										AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
									}
								}

								//jacobian.
								Jacobian(AuxNeuronState_p, jacnum, index);

								for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
									for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
										J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * this->BifixedElapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
										if (z == t){
											J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
										}
									}
								}

								this->invermat(J, inv_J);

								for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
									float aux = 0.0;
									for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
										aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
									}
									AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
								}

								//We calculate the difference between both aproximations.
								float aux = 0.0f;
								float aux2 = 0.0f;
								for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
									aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
									if (aux > aux2){
										aux2 = aux;
									}
								}

								memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

								epsi = aux2;
								k++;
							}

							//We increase the state of the integration method.
							if (state[index] < BDForder){
								state[index]++;
							}


							//We acumulate these new values for the next step.
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){

								for (int i = (state[index] - 1); i > 0; i--){
									D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
								D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
								for (int i = 1; i < state[index]; i++){
									D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
							}

							if (state[index]>1){
								for (int i = state[index] - 2; i > 0; i--){
									memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
								}

								memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
							}
							memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



							//Finaly, we evaluate the neural state variables with time dependence.
							this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 1);

							//Update the last spike time.
							this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, this->BifixedElapsedTimeInSeconds);

							this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);

							//Acumulate the membrane potential in a variable
							this->IncrementValidIntegrationVariable(NeuronState[0]);

							//Set the CoeficientSelector to 1.
							CoeficientSelector[index] = 1;
							PreviousIntegrationStep[index] = this->BifixedElapsedTimeInNeuronModelScale;

							if (NeuronState[0] > this->startVoltageThreshold && this->integrationMethodState[index] == 1){
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
									CoeficientSelector[index] = 2;
								}
							}
						}
						if (this->integrationMethodState[index] == 1){
							this->integrationMethodState[index] = 0;
							CoeficientSelector[index] = 2;
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
			out << "\t\tIntegration Method Type: " << Bifixed_BDFn::GetName() << endl;
			out << "\t\tBDF order: " << this->BDForder << endl;
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
			if (BDForder>1){
				PreviousNeuronState = (float **)new float*[BDForder - 1];
				for (int i = 0; i<(BDForder - 1); i++){
					PreviousNeuronState[i] = new float[N_neurons*this->neuron_model->N_DifferentialNeuronState];
				}
			}
			D = (float **)new float*[BDForder];
			for (int i = 0; i<BDForder; i++){
				D[i] = new float[N_neurons*this->neuron_model->N_DifferentialNeuronState];
			}

			state = new int[N_neurons]();


			PreviousIntegrationStep = new float[N_neurons]();
			CoeficientSelector = new int[N_neurons];

			for (int i = 0; i<N_neurons; i++){
				CoeficientSelector[i] = 1;
			}


			this->integrationMethodCounter = new int[N_neurons]();
			this->integrationMethodState = new int[N_neurons]();



			float decrement = this->BifixedElapsedTimeInNeuronModelScale / this->elapsedTimeInNeuronModelScale;
			float increment = this->elapsedTimeInNeuronModelScale / this->BifixedElapsedTimeInNeuronModelScale;

			//decrement of h
			Coeficient[0][0][0] = 1.0f;
			Coeficient[0][0][1] = 0.0f;
			Coeficient[0][0][2] = 1.0f;
			Coeficient[0][0][3] = 0.0f;
			Coeficient[0][1][0] = 1.0f;
			Coeficient[0][1][1] = 0.0f;
			Coeficient[0][1][2] = 1.0f;
			Coeficient[0][1][3] = 0.0f;
			Coeficient[0][2][0] = 2.0f / 3.0f; //beta 0
			Coeficient[0][2][1] = (decrement - 1) / 3.0f; //beta 1
			Coeficient[0][2][2] = 1 + decrement*decrement / 3.0f;//alpha 0
			Coeficient[0][2][3] = -decrement*decrement / 3.0f;//alpha 1

			//Equal h;
			Coeficient[1][0][0] = 1.0f;
			Coeficient[1][0][1] = 0.0f;
			Coeficient[1][0][2] = 1.0f;
			Coeficient[1][0][3] = 0.0f;
			Coeficient[1][1][0] = 1.0f;
			Coeficient[1][1][1] = 0.0f;
			Coeficient[1][1][2] = 1.0f;
			Coeficient[1][1][3] = 0.0f;
			Coeficient[1][2][0] = 2.0f / 3.0f;
			Coeficient[1][2][1] = 0.0f;
			Coeficient[1][2][2] = 4.0f / 3.0f;
			Coeficient[1][2][3] = -1.0f / 3.0f;

			//increment of h
			Coeficient[2][0][0] = 1.0f;
			Coeficient[2][0][1] = 0.0f;
			Coeficient[2][0][2] = 1.0f;
			Coeficient[2][0][3] = 0.0f;
			Coeficient[2][1][0] = 1.0f;
			Coeficient[2][1][1] = 0.0f;
			Coeficient[2][1][2] = 1.0f;
			Coeficient[2][1][3] = 0.0f;
			Coeficient[2][2][0] = 2.0f / 3.0f;
			Coeficient[2][2][1] = (increment - 1) / 3.0f;
			Coeficient[2][2][2] = 1 + increment*increment / 3.0f;
			Coeficient[2][2][3] = -increment*increment / 3.0f;
		}

		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param index indicate which neuron must be reseted.
		 */
		void resetState(int index){
			state[index] = 0;
		}

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
		}

		 /*!
		 * \brief It calculate numerically the Jacobian .
		 *
		 * It calculate numerically the Jacobian.
		 *
		 * \param NeuronState neuron state variables of one neuron.
		 * \param jancum vector where is stored the Jacobian.
		 * \param index neuron state index.
		 */
		void Jacobian(float * NeuronState, float * jacnum, int index){
			float epsi = this->elapsedTimeInNeuronModelScale * 0.1f;
			float inv_epsi = 1.0f / epsi;
			float JacAuxNeuronState[MAX_VARIABLES];
			float JacAuxNeuronState_pos[MAX_VARIABLES];
			float JacAuxNeuronState_neg[MAX_VARIABLES];

			memcpy(JacAuxNeuronState, NeuronState, sizeof(float)*this->neuron_model->N_NeuronStateVariables);
			this->neuron_model->EvaluateDifferentialEquation(JacAuxNeuronState, JacAuxNeuronState_pos, index, this->elapsedTimeInNeuronModelScale);

			for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				memcpy(JacAuxNeuronState, NeuronState, sizeof(float)*this->neuron_model->N_NeuronStateVariables);

				JacAuxNeuronState[j] -= epsi;
				this->neuron_model->EvaluateDifferentialEquation(JacAuxNeuronState, JacAuxNeuronState_neg, index, this->elapsedTimeInNeuronModelScale);

				for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
					jacnum[z*this->neuron_model->N_DifferentialNeuronState + j] = (JacAuxNeuronState_pos[z] - JacAuxNeuronState_neg[z])*inv_epsi;
				}
			}
		}

		 /*!
		 * \brief It calculate the inverse of a square matrix using Gauss-Jordan Method.
		 *
		 * It calculate the inverse of a square matrix using Gauss-Jordan Method.
		 *
		 * \param a pointer to the square matrix.
		 * \param ainv pointer where inverse of the square matrix will be stored.
		 */
		void invermat(float *a, float *ainv){
			if (this->neuron_model->N_DifferentialNeuronState == 1){
				ainv[0] = 1.0f / a[0];
			}
			else{
				float coef, element, inv_element;
				int i, j, s;

				float local_a[MAX_VARIABLES*MAX_VARIABLES];
				float local_ainv[MAX_VARIABLES*MAX_VARIABLES] = {};

				memcpy(local_a, a, sizeof(float)*this->neuron_model->N_DifferentialNeuronState*this->neuron_model->N_DifferentialNeuronState);
				for (i = 0; i<this->neuron_model->N_DifferentialNeuronState; i++){
					local_ainv[i*this->neuron_model->N_DifferentialNeuronState + i] = 1.0f;
				}

				//Iteraciones
				for (s = 0; s<this->neuron_model->N_DifferentialNeuronState; s++)
				{
					element = local_a[s*this->neuron_model->N_DifferentialNeuronState + s];

					if (element == 0){
						for (int n = s + 1; n<this->neuron_model->N_DifferentialNeuronState; n++){
							element = local_a[n*this->neuron_model->N_DifferentialNeuronState + s];
							if (element != 0){
								for (int m = 0; m<this->neuron_model->N_DifferentialNeuronState; m++){
									float value = local_a[n*this->neuron_model->N_DifferentialNeuronState + m];
									local_a[n*this->neuron_model->N_DifferentialNeuronState + m] = local_a[s*this->neuron_model->N_DifferentialNeuronState + m];
									local_a[s*this->neuron_model->N_DifferentialNeuronState + m] = value;

									value = local_ainv[n*this->neuron_model->N_DifferentialNeuronState + m];
									local_ainv[n*this->neuron_model->N_DifferentialNeuronState + m] = local_ainv[s*this->neuron_model->N_DifferentialNeuronState + m];
									local_ainv[s*this->neuron_model->N_DifferentialNeuronState + m] = value;
								}
								break;
							}
							if (n == (this->neuron_model->N_DifferentialNeuronState - 1)){
								printf("This matrix is not invertible\n");
								exit(0);
							}

						}
					}

					inv_element = 1.0f / element;
					for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						local_a[s*this->neuron_model->N_DifferentialNeuronState + j] *= inv_element;
						local_ainv[s*this->neuron_model->N_DifferentialNeuronState + j] *= inv_element;
					}

					for (i = 0; i<this->neuron_model->N_DifferentialNeuronState; i++)
					{
						if (i != s){
							coef = -local_a[i*this->neuron_model->N_DifferentialNeuronState + s];
							for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
								local_a[i*this->neuron_model->N_DifferentialNeuronState + j] += local_a[s*this->neuron_model->N_DifferentialNeuronState + j] * coef;
								local_ainv[i*this->neuron_model->N_DifferentialNeuronState + j] += local_ainv[s*this->neuron_model->N_DifferentialNeuronState + j] * coef;
							}
						}
					}
				}
				memcpy(ainv, local_ainv, sizeof(float)*this->neuron_model->N_DifferentialNeuronState*this->neuron_model->N_DifferentialNeuronState);
			}
		}


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
			newMap["bdf_order"] = this->BDForder;
			newMap["name"] = Bifixed_BDFn::GetName();
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
			std::map<std::string, boost::any>::iterator it = param_map.find("bdf_order");
			if (it != param_map.end()){
				int new_BDF_Order = boost::any_cast<int>(it->second);
				if (new_BDF_Order < 1 || new_BDF_Order > 2){
					throw EDLUTException(TASK_BDF_ORDER_LOAD, ERROR_BDF_ORDER_VALUE, REPAIR_BDF_ORDER);
				}
				this->BDForder = new_BDF_Order;
				param_map.erase(it);
			}
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
			newMap["name"] = Bifixed_BDFn::GetName();
			newMap["bdf_order"] = 2;
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
			skip_comments(fh, Currentline);
			int BDF_Order;
			// Load BDF integration method order
			if (fscanf(fh, "%d", &BDF_Order) != 1) {
				throw EDLUTException(TASK_BDF_ORDER_LOAD, ERROR_BDF_ORDER_READ, REPAIR_BDF_ORDER);
			}
			if (BDF_Order < 1 || BDF_Order > 2){
				throw EDLUTException(TASK_BDF_ORDER_LOAD, ERROR_BDF_ORDER_VALUE, REPAIR_BDF_ORDER);
			}

			//load integration time step.
			ModelDescription nmodel = BifixedStep<Neuron_Model>::ParseIntegrationMethod(fh, Currentline);

			nmodel.param_map["bdf_order"] = BDF_Order;
		    nmodel.model_name = Bifixed_BDFn::GetName();
			return nmodel;
		};

		/*!
		 * \brief It returns the name of the integration method
		 *
		 * It returns the name of the integration method
		 */
		static std::string GetName(){
			return "Bifixed_BDF";
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
			Bifixed_BDFn * newmodel = new Bifixed_BDFn(nmodel);
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
            const Bifixed_BDFn * e = dynamic_cast<const Bifixed_BDFn *> (rhs);
            if (e == 0) return false;
            return this->BDForder==e->BDForder;
        };

};

#endif /* BIFIXED_BDFn_H_ */
