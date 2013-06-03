/***************************************************************************
 *                           EdidioGranuleCell_TimeDriven.h                *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef EGIDIOGRANULECELL_TIMEDRIVEN_H_
#define EGIDIOGRANULECELL_TIMEDRIVEN_H_


/*!
 * \file EdidioGranuleCell_TimeDriven.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model for a cerebellar 
 * granule cell. This neuron model has 15 differential equations and 2 time dependent equations (conductances).
 */

#include "./TimeDrivenNeuronModel.h"

#include <string>

using namespace std;

class InputSpike;
class VectorNeuronState;
class Interconnection;

 

/*!
 * \class EdidioGranuleCell_TimeDriven
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model for a cerebellar granule cell.
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class EgidioGranuleCell_TimeDriven : public TimeDrivenNeuronModel {
	protected:

		//const float gMAXNa_f=0.013;
		//const float gMAXNa_r=0.0005;
		//const float gMAXNa_p=2.0e-5;
		//const float gMAXK_V=0.003;
		//const float gMAXK_A=0.004;
		//const float gMAXK_IR=0.0009;
		//const float gMAXK_Ca=0.004;
		//const float gMAXCa=0.00046;
		//const float gMAXK_sl=0.00035;

		float gMAXNa_f;
		float gMAXNa_r;
		float gMAXNa_p;
		float gMAXK_V;
		float gMAXK_A;
		float gMAXK_IR;
		float gMAXK_Ca;
		float gMAXCa;
		float gMAXK_sl;

		const float gLkg1;
		const float gLkg2;
		const float VNa;
		const float VK;
		const float VLkg1;
		const float VLkg2;
		const float V0_xK_Ai;
		const float K_xK_Ai;
		const float V0_yK_Ai;
		const float K_yK_Ai;
		const float V0_xK_sli;
		const float B_xK_sli;
		const float F;
		const float A;
		const float d;
		const float betaCa;
		const float Ca0;
		const float R;
		const float cao;
		//Membrane Capacitance = 1uF/cm^2 = 10^-3mF/cm^2;
		const float Cm;

		const float temper;

		// gating-kinetic correction for experiment temperature=20ºC;
		const float Q10_20;
		// gating-kinetic correction for experiment temperature=22ºC;
		const float Q10_22;
		// experiment temperature=30ºC;
		const float Q10_30;
		// experiment temperature=6.3ºC;
		const float Q10_6_3;

		const float I_inj_abs;

		// Injected Current in absolute terms = -10pA;
		// Cell membrane area = pi*2*r*L = pi*9.76*9.76 = 299.26058um^2 = 299.26058*10^-8cm^2;
		// Injected current per area unit = -I_inj_abs/ 299.26058*10^-8cm^2 = I_inj;
		const float I_inj;

		const float eexc;
		const float einh;
		const float texc;
		const float tinh;
		const float vthr;


		/*!
		 * \brief It loads the neuron model description.
		 *
		 * It loads the neuron type description from the file .cfg.
		 *
		 * \param ConfigFile Name of the neuron description file (*.cfg).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel(string ConfigFile) throw (EDLUTFileException);


		/*!
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *
		 * \param index The cell index inside the VectorNeuronState.
		 * \param State Cell current state.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		virtual void SynapsisEffect(int index, VectorNeuronState * State, Interconnection * InputConnection);


		/*!
		 * \brief 
		 *
		 * 
		 *
		 * \param ci.
		 * \param co.
		 * \param z.
		 * \param temper temperature.
		 */
		float nernst(float ci, float co, float z, float temper);


		/*!
		 * \brief 
		 *
		 * 
		 *
		 * \param x.
		 * \param y.
		 */
		float linoid(float x, float y);


	public:

		/*!
		 * \brief Number of state variables for each cell.
		*/
		static const int N_NeuronStateVariables=17;

		/*!
		 * \brief Number of state variables witch are calculate with a differential equation for each cell.
		*/
		static const int N_DifferentialNeuronState=15;

		/*!
		 * \brief Number of state variables witch are calculate with a time dependent equation for each cell.
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
		EgidioGranuleCell_TimeDriven(string NeuronTypeID, string NeuronModelID);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~EgidioGranuleCell_TimeDriven();


		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 */
		virtual void LoadNeuronModel() throw (EDLUTFileException);


		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
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
		 * \param InputSpike The spike happened.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessInputSpike(PropagatedSpike *  InputSpike);


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
		virtual bool UpdateState(int index, VectorNeuronState * State, double CurrentTime);


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
		virtual void InitializeStates(int N_neurons);


		/*!
		 * \brief It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * \param NeuronState value of the neuron state variables where differential equations are evaluated.
		 * \param AuxNeuronState results of the differential equations evaluation.
		 */
		virtual void EvaluateDifferentialEcuation(float * NeuronState, float * AuxNeuronState);


		/*!
		 * \brief It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * \param NeuronState value of the neuron state variables where time dependent equations are evaluated.
		 * \param elapsed_time integration time step.
		 */
		virtual void EvaluateTimeDependentEcuation(float * NeuronState, double elapsed_time);
};

#endif /* EGIDIOGRANULECELL_TIMEDRIVEN_H_ */
