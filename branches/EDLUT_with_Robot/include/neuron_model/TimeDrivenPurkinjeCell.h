/***************************************************************************
 *                           TimeDrivenPurkinjeCell.h                      *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Richard Carrill, Niceto Luque and    *
						  Francisco Naveros								   *
 * email                : rcarrillo@ugr.es, nluque@ugr.es and			   *
						  fnaveros@atc.ugr.es							   *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TIMEDRIVENPURKINJECELL_H_
#define TIMEDRIVENPURKINJECELL_H_

/*!
 * \file TimeDrivenPurkinjeCell.h
 *
 * \author Richard Carrillo
 * \author Niceto Luque
 * \author Francisco Naveros
 * \date May 2015
 *
 * This file declares a class which implement a Purkinje cell model.
 */

#include "./TimeDrivenNeuronModel.h"
#include <cmath>
#include <string>

using namespace std;

class InputSpike;
class VectorNeuronState;
class Interconnection;



/*!
 * \class TimeDrivenPurkinjeCell.h
 *
 * \brief Time driven neuron model with a membrane potential, two current channels and two conductances.
 *
 * \author Richard Carrillo
 * \author Niceto Luque
 * \author Francisco Naveros
 * \date May 2015
 */
class TimeDrivenPurkinjeCell : public TimeDrivenNeuronModel {
	protected:
		/*!
		 * \brief channel values
		 * alpha_ca
		 * inv_tau_ca
		 * alpha_M
		 * inv_tau_M
		 */
		static float * channel_values;
		static const float Max_V;
		static const float Min_V;
		static const int TableSize=1024*1024;
		static const float aux;
		static const int N_variables=4;

		/*!
		 * \brief leak current 
		 */
		const float g_L;

		/*!
		 * \brief high-threshold noninactivating calcium current 
		 */
		const float g_Ca;

		/*!
		 * \brief muscarinic receptor suppressed potassium current (or M current) 
		 */
		const float g_M;

		
		/*!
		 * \brief Cylinder length of the soma (cm)
		 */
		const float Cylinder_length_of_the_soma;

				
		/*!
		 * \brief Radius of the soma (cm)
		 */
		const float Radius_of_the_soma;
					
		/*!
		 * \brief Cell area (cm^2)
		 */
		const float Area;
		const float inv_Area;
					
		/*!
		 * \brief Membrane capacitance(uF/cm^2)
		 */
		const float Membrane_capacitance;
		const float inv_Membrane_capacitance;



		/*!
		 * \brief Excitatory reversal potential (mV)
		 */
		float eexc;

		/*!
		 * \brief Inhibitory reversal potential (mV)
		 */
		float einh;

		/*!
		 * \brief Firing threshold (mV)
		 */
		float vthr;

		/*!
		 * \brief Resting potential (mV)
		 */
		float erest;

		/*!
		 * \brief AMPA receptor time constant (ms)
		 */
		float texc;
		float inv_texc;

		/*!
		 * \brief GABA receptor time constant (ms)
		 */
		float tinh;
		float inv_tinh;


		/*!
		 * \brief Refractory period (ms)
		 */
		float tref;
		float tref_0_5;
		float inv_tref_0_5;

		/*!
		 * \brief Peak amplitude (mV)
		 */
		float spkpeak;




		/*!
		 * \brief It loads the neuron model description.
		 *
		 * It loads the neuron type description from the file .cfg.
		 *
		 * \param ConfigFile Name of the neuron description file (*.cfg).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		void LoadNeuronModel(string ConfigFile) throw (EDLUTFileException);


		/*!
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *
		 * \param index The cell index inside the VectorNeuronState.
		 * \param State Cell current state.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		void SynapsisEffect(int index, Interconnection * InputConnection);



	public:

		/*!
		 * \brief Number of state variables for each cell.
		*/
		static const int N_NeuronStateVariables=5;

		/*!
		 * \brief Number of state variables witch are calculate with a differential equation for each cell (V, ca and M).
		*/
		static const int N_DifferentialNeuronState=3;

		/*!
		 * \brief Number of state variables witch are calculate with a time dependent equation for each cell (g_exc and g_inh).
		*/
		static const int N_TimeDependentNeuronState=2;


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model identificator.
		 * \param NeuronModelID Neuron model configuration file.
		 */
		TimeDrivenPurkinjeCell(string NeuronTypeID, string NeuronModelID);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenPurkinjeCell();


		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
 		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel() throw (EDLUTFileException);


		/*!
		 * \brief It return the Neuron Model VectorNeuronState 
		 *
		 * It return the Neuron Model VectorNeuronState 
		 *
		 */
		virtual VectorNeuronState * InitializeState();


		/*!
		 * \brief It processes a propagated spike (input spike in the cell).
		 *
		 * It processes a propagated spike (input spike in the cell).
		 *
		 * \note This function doesn't generate the next propagated spike. It must be externally done.
		 *
		 * \param inter the interconection which propagate the spike
		 * \param target the neuron which receives the spike
		 * \param time the time of the spike.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessInputSpike(Interconnection * inter, Neuron * target, double time);


		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the VectorNeuronState. if index=-1, updating all cell.
		 * \param The current neuron state.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(int index, double CurrentTime);



		/*!
		 * \brief It prints the time-driven model info.
		 *
		 * It prints the current time-driven model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);


		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex);


		/*!
		 * \brief It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * \param NeuronState value of the neuron state variables where differential equations are evaluated.
		 * \param AuxNeuronState results of the differential equations evaluation.
		 */
		virtual void EvaluateDifferentialEcuation(float * NeuronState, float * AuxNeuronState, int index);


		/*!
		 * \brief It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * \param NeuronState value of the neuron state variables where time dependent equations are evaluated.
		 * \param elapsed_time integration time step.
		 */
		virtual void EvaluateTimeDependentEcuation(float * NeuronState, float elapsed_time);


		/*!
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return A a valid connection type for this neuron model.
		 */
		virtual int CheckSynapseTypeNumber(int Type);


		static float * Generate_channel_values(){
			float * NewLookUpTable=new float[TableSize*N_variables];
			for(int i=0; i<TableSize; i++){
				float V = Min_V + ((Max_V-Min_V)*i)/(TableSize-1);
				
				//alpha_ca
				float alpha_ca=1.6f/(1+exp(-0.072f*(V-5.0f)));
				NewLookUpTable[i*N_variables]=alpha_ca;
				
				//inv_tau_ca
				float beta_ca=(0.02f*(V+8.9f))/(exp((V+8.9f)*0.2f)-1.0f);
				float inv_tau_ca=alpha_ca+beta_ca;
				NewLookUpTable[i*N_variables+1]=inv_tau_ca;
				
				//alpha_M
				float alpha_M=0.3f/(1+exp((-V-2.0f)*0.2f));
				NewLookUpTable[i*N_variables+2]=alpha_M;
				
				//inv_tau_M
				float beta_M=0.001f*exp((-V-60.0f)*0.055555555555555f);
				float inv_tau_M=alpha_M+beta_M;
				NewLookUpTable[i*N_variables+3]=inv_tau_M;
			}
			return NewLookUpTable;
		}


		static float * Get_channel_values(float value){
				int position=int((value-Min_V)*aux);
				return (channel_values + position*N_variables);
		} 	

};

#endif /* TIMEDRIVENPURKINJECELL_H_ */
