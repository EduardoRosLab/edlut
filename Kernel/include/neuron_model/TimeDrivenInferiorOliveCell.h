/***************************************************************************
 *                           TimeDrivenInferiorOliveCell.h                 *
 *                           -------------------                           *
 * copyright            : (C) 2019 by Niceto Luque and Francisco Naveros   *
 * email                : nluque@ugr.es and	fnaveros@ugr.es  			   *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TIMEDRIVENINFERIOROLIVECELL_H_
#define TIMEDRIVENINFERIOROLIVECELL_H_

/*!
 * \file TimeDrivenInferiorOliveCell.h
 *
  * \author Niceto Luque
 * \author Francisco Naveros
 * \date November 2019
 *
 * This file declares a class which implements an Inferior Olive cell as a Leaky Integrate-And-Fire (LIF) neuron model with one 
 * differential equation (membrane potential), three time dependent equations (excitatory, inhibitory and
 * NMDA conductances), one external input current synapse, and one electrical coupling synapse.
 */

#include "neuron_model/TimeDrivenNeuronModel.h"

#include <string>

using namespace std;


class VectorNeuronState;
class InternalSpike;
class Interconnection;
struct ModelDescription;


/*!
 * \class TimeDrivenInferiorOliveCell.h
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model with a membrane potential (V), three
 * conductances (excitatory, inhibitory and NMDA conductances), one external input current synapse,
 * and electrical coupling synapse.
 *
 * \author Niceto Luque
 * \author Francisco Naveros
 * \date November 2019
 */
class TimeDrivenInferiorOliveCell : public TimeDrivenNeuronModel {
	protected:
	
		/*!
		 * \brief Spike peak amplitude (mV)
		 */
		const float spk_peak;
	
		/*!
		 * \brief Excitatory reversal potential (mV)
		 */
		float e_exc;

		/*!
		 * \brief Inhibitory reversal potential (mV)
		 */
		float e_inh;

		/*!
		 * \brief Resting potential (mV)
		 */
		float e_leak;

		/*!
		* \brief Firing threshold (mV)
		*/
		float v_thr;

		/*!
		 * \brief Membrane capacitance(uF/cm^2)
		 */
		float c_m;
		float inv_c_m;

		/*!
		 * \brief AMPA receptor time constant (ms)
		 */
		float tau_exc;
		float inv_tau_exc;

		/*!
		 * \brief GABA receptor time constant (ms)
		 */
		float tau_inh;
		float inv_tau_inh;
		
		/*!
		 * \brief NMDA receptor time constant (ms)
		 */
		float tau_nmda;
		float inv_tau_nmda;

		/*!
		 * \brief Refractory period (ms)
		 */
		float tau_ref;
		float tau_ref_0_5;
		float inv_tau_ref_0_5;
		
		/*!
		 * \brief Resting conductance current (mS/cm^2)
		 */
		float g_leak;
		
		/*!
		 * \brief Cell area (cm^2)
		 */
		float area;
		float inv_area; 


		/*!
		 * \brief Number of compling synapses
		 */
		int * N_coupling_synapses;
		
		/*!
		 * \brief Index of compling synapses
		 */
		int * index_coupling_synapses;
		
		/*!
		 * \brief Pointer to synaptic weight of coupling synapses
		 */
		float ** Weight_coupling_synapses;
		
		/*!
		 * \brief Pointer to membrane potential of target neurons in coupling synapses
		 */
		float *** Potential_coupling_synapses;
		
		
		/*!
		* \brief It computest the g_nmda_inf values based on the e_exc and e_inh values.
		*
		* It computest the g_nmda_inf values based on the e_exc and e_inh values.
		*/
		virtual void Generate_g_nmda_inf_values();

		/*!
		* \brief It returns the g_nmda_value corresponding with the membrane potential (V_m).
		*
		* It returns the g_nmda_value corresponding with the membrane potential (V_m).
		*
		* \param V_m membrane potential.
		*
		* \return g_nmda_value corresponding with the membrane potential.
		*/
		virtual float Get_g_nmda_inf(float V_m);

		/*!
		* \brief It initializes the CurrentSynapsis object.
		*
		* It initializes the CurrentSynapsis object.
		*/
		virtual void InitializeCurrentSynapsis(int N_neurons);


	public:

		/*!
		 * \brief Number of state variables for each cell.
		*/
		const int N_NeuronStateVariables=6;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		 */
		const int N_DifferentialNeuronState = 1;
			//Index of each differential neuron state variable
			const int V_m_index = 0;

		/*!
		 * \brief Number of state variables which are calculate with a time dependent equation for each cell (EXC, INH, NMDA, EXT_I, I_COU).
		 */
		const int N_TimeDependentNeuronState = 5;
			//Index of each time dependent neuron state variable
			const int EXC_index = N_DifferentialNeuronState;
			const int INH_index = N_DifferentialNeuronState + 1;
			const int NMDA_index = N_DifferentialNeuronState + 2;
			const int EXT_I_index = N_DifferentialNeuronState + 3;
			const int I_COU_index = N_DifferentialNeuronState + 4;

		/*!
	 	 * \brief Boolean variable setting in runtime if the neuron model receive each one of the supported input synpase types (N_TimeDependentNeuronState) 
		 */
		bool EXC, INH, NMDA, EXT_I, I_COU;

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 */
		TimeDrivenInferiorOliveCell();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenInferiorOliveCell();

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
		 * \param time the time of the spike.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessInputSpike(Interconnection * inter, double time);

		/*!
		* \brief It processes a propagated current (input current in the cell).
		*
		* It processes a propagated current (input current in the cell).
		*
		* \param inter the interconection which propagate the spike
		* \param target the neuron which receives the spike
		* \param Current input current.
		*/
		virtual void ProcessInputCurrent(Interconnection * inter, Neuron * target, float current);

		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the VectorNeuronState. if index=-1, updating all cell.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(int index, double CurrentTime);

		/*!
		 * \brief It gets the neuron output activity type (spikes or currents).
		 *
		 * It gets the neuron output activity type (spikes or currents).
		 *
		 * \return The neuron output activity type (spikes or currents).
		 */
		enum NeuronModelOutputActivityType GetModelOutputActivityType();

		/*!
		 * \brief It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * \return The neuron input activity types (spikes and/or currents or none).
		 */
		enum NeuronModelInputActivityType GetModelInputActivityType();

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
		 * \param OpenMPQueueIndex openmp index
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex);

		/*!
		 * \brief It evaluates if a neuron must spike.
		 *
		 * It evaluates if a neuron must spike.
		 *
		 * \param previous_V previous membrane potential
		 * \param NeuronState neuron state variables.
		 * \param index Neuron index inside the neuron model.
		 * \param elapsedTimeInNeuronModelScale integration method step.
		 * \return It returns if a neuron must spike.
		 */
		void EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale);

		/*!
		 * \brief It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * \param NeuronState value of the neuron state variables where differential equations are evaluated.
		 * \param AuxNeuronState results of the differential equations evaluation.
		 * \param index Neuron index inside the VectorNeuronState
		 */
		void EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time);

		/*!
		 * \brief It evaluates the time depedendent Equation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * It evaluates the time depedendent Equation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * \param NeuronState value of the neuron state variables where time dependent equations are evaluated.
		 * \param elapsed_time integration time step.
		 * \param elapsed_time_index index inside the conductance_exp_values array.
		 */
		void EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index);

		/*!
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Conncetion input connection type.
		 *
		 * \return If the neuron model supports this connection type
		 */
		virtual bool CheckSynapseType(Interconnection * connection);

		/*!
		 * \brief It calculates the conductace exponential value for an elapsed time.
		 *
		 * It calculates the conductace exponential value for an elapsed time.
		 *
		 * \param index elapsed time index .
		 * \param elapses_time elapsed time.
		 */
		void Calculate_conductance_exp_values(int index, float elapsed_time);

		/*!
		* \brief It calculates the number of electrical coupling synapses.
		*
		* It calculates the number for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		void CalculateElectricalCouplingSynapseNumber(Interconnection * inter);

		/*!
		* \brief It allocate memory for electrical coupling synapse dependencies.
		*
		* It allocate memory for electrical coupling synapse dependencies.
		*/
		void InitializeElectricalCouplingSynapseDependencies();

		/*!
		* \brief It calculates the dependencies for electrical coupling synapses.
		*
		* It calculates the dependencies for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		void CalculateElectricalCouplingSynapseDependencies(Interconnection * inter);
		
		/*!
		 * \brief It gets the required parameter in the adaptative integration methods (Bifixed_Euler, Bifixed_RK2, Bifixed_RK4, Bifixed_BDF1 and Bifixed_BDF2).
		 *
		 * It gets the required parameter in the adaptative integration methods (Bifixed_Euler, Bifixed_RK2, Bifixed_RK4, Bifixed_BDF1 and Bifixed_BDF2).
		 *
		 * \param startVoltageThreshold, when the membrane potential reaches this value, the multi-step integration methods change the integration
		 *  step from elapsedTimeInNeuronModelScale to bifixedElapsedTimeInNeuronModelScale.
		 * \param endVoltageThreshold, when the membrane potential reaches this value, the multi-step integration methods change the integration
		 *  step from bifixedElapsedTimeInNeuronModelScale to ElapsedTimeInNeuronModelScale after timeAfterEndVoltageThreshold in seconds.
		 * \param timeAfterEndVoltageThreshold, time in seconds that the multi-step integration methods maintain the bifixedElapsedTimeInNeuronModelScale
		 *  after the endVoltageThreshold
		 */
		virtual void GetBifixedStepParameters(float & startVoltageThreshold, float & endVoltageThreshold, float & timeAfterEndVoltageThreshold);

		/*!
		 * \brief It returns the neuron model parameters.
		 *
		 * It returns the neuron model parameters.
		 *
		 * \returns A dictionary with the neuron model parameters
		 */
		virtual std::map<std::string,boost::any> GetParameters() const;

		/*!
		* \brief It returns the neuron model parameters for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* It returns the neuron model parameters for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* \param index neuron index inside the neuron model.
		*
		* \returns A dictionary with the neuron model parameters
		*
		* NOTE: this function is accesible throgh the Simulatiob_API interface.
		*/
		virtual std::map<std::string, boost::any> GetSpecificNeuronParameters(int index) const noexcept(false);

		/*!
		 * \brief It loads the neuron model properties.
		 *
		 * It loads the neuron model properties from parameter map.
		 *
		 * \param param_map The dictionary with the neuron model parameters.
		 *
		 * \throw EDLUTException If it happens a mistake with the parameters in the dictionary.
		 */
		virtual void SetParameters(std::map<std::string, boost::any> param_map) noexcept(false);

		/*!
		* \brief It creates the integration method
		*
		* It creates the integration methods using the parameter map.
		*
		* \param param_map The dictionary with the integration method parameters.
		*
		* \throw EDLUTException If it happens a mistake with the parameters in the dictionary.
		*/
		virtual IntegrationMethod * CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false);

		/*!
		 * \brief It returns the default parameters of the neuron model.
		 *
		 * It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		 * set for this neuron model.
		 *
		 * \returns A dictionary with the neuron model default parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();

		/*!
		 * \brief It creates a new neuron model object of this type.
		 *
		 * It creates a new neuron model object of this type.
		 *
		 * \param param_map The neuron model description object.
		 *
		 * \return A newly created NeuronModel object.
		 */
		static NeuronModel* CreateNeuronModel(ModelDescription nmDescription);

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 *
		 * \param FileName This parameter is not used. It is stub parameter for homegeneity with other neuron models.
		 *
		 * \return A neuron model description object with the parameters of the neuron model.
		 */
		static ModelDescription ParseNeuronModel(std::string FileName) noexcept(false);

		/*!
		 * \brief It returns the name of the neuron type
		 *
		 * It returns the name of the neuron type.
		 */
		static std::string GetName();

		/*!
		* \brief It returns the neuron model information, including its parameters.
		*
		* It returns the neuron model information, including its parameters.
		*
		*\return a map with the neuron model information, including its parameters.
		*/
		static std::map<std::string, std::string> GetNeuronModelInfo();

        /*!
         * \brief Comparison operator between neuron models.
         *
         * It compares two neuron models.
         *
         * \return True if the neuron models are of the same type and with the same parameters.
         */
        virtual bool compare(const NeuronModel * rhs) const{
            if (!TimeDrivenNeuronModel::compare(rhs)){
                return false;
            }
            const TimeDrivenInferiorOliveCell * e = dynamic_cast<const TimeDrivenInferiorOliveCell *> (rhs);
            if (e == 0) return false;

			return this->e_exc == e->e_exc &&
				this->e_inh == e->e_inh &&
				this->e_leak == e->e_leak &&
				this->v_thr == e->v_thr &&
				this->c_m == e->c_m &&
				this->tau_exc == e->tau_exc &&
				this->tau_inh == e->tau_inh &&
				this->tau_nmda==e->tau_nmda &&
				this->tau_ref == e->tau_ref &&
				this->g_leak == e->g_leak &&
				this->area == e->area;
        };
};

#endif /* TIMEDRIVENINFERIOROLIVECELL_H_ */
