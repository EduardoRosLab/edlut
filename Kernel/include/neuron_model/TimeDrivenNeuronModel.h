/***************************************************************************
 *                           TimeDrivenNeuronModel.h                       *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@ugr.es             *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TIMEDRIVENNEURONMODEL_H_
#define TIMEDRIVENNEURONMODEL_H_

/*!
 * \file TimeDrivenNeuronModel.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date January 2011
 *
 * This file declares a class which abstracts an time-driven neuron model in a CPU.
 */

#include "neuron_model/TimeDrivenModel.h"
#include "neuron_model/CurrentSynapseModel.h"

#include "integration_method/IntegrationMethod.h"
#include "integration_method/IntegrationMethodFactory.h"
#include "simulation/NetworkDescription.h"
//#include "integration_method/Euler.h"
//#include "integration_method/RK2.h"
//#include "integration_method/RK4.h"
//#include "integration_method/BDFn.h"

//#include "integration_method/Bifixed_Euler.h"
//#include "integration_method/Bifixed_RK2.h"
//#include "integration_method/Bifixed_RK4.h"
//#include "integration_method/Bifixed_BDFn.h"
//#include "integration_method/FixedStepSRM.h"

//#include "simulation/Utils.h"
//#include "simulation/Configuration.h"



#include <cstring>
//#include <cstdlib>

using namespace std;

#include "simulation/Configuration.h"
class VectorNeuronState;
class InternalSpike;
class Interconnection;
//struct ModelDescription;



/*!
 * \class TimeDrivenNeuronModel
 *
 * \brief Time-Driven Spiking neuron model in a CPU
 *
 * This class abstracts the behavior of time-driven neuron models in spiking neural
 * networks implemented in CPU.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date January 2011
 */
class TimeDrivenNeuronModel : public TimeDrivenModel {
	protected:


		/*!
		 * \brief integration method.
		*/
		IntegrationMethod * integration_method;

		/*!
		 * \brief Auxiliar array for time dependente variables.
		*/
		float * conductance_exp_values;

		/*!
		 * \brief Auxiliar variable for time dependente variables.
		*/
		int N_conductances;


		/*!
		* \brief Precomputed look-up table of "TableSize" elements storing the value of g_nmda_inf variable inf function of differnt membrane potential values.
		*/
		static const int TableSizeNMDA = 256;
		float auxNMDA;
		float g_nmda_inf_values[TableSizeNMDA];

		/*!
		* \brief Object to store the input currents of synapses that receive currents.
		*/
		CurrentSynapseModel * CurrentSynapsis;

		/*!
		* \brief It computest the g_nmda_inf values based on the e_exc and e_inh values.
		*
		* It computest the g_nmda_inf values based on the e_exc and e_inh values.
		*/
		virtual void Generate_g_nmda_inf_values() = 0;

		/*!
		* \brief It returns the g_nmda_value corresponding with the membrane potential (V_m).
		*
		* It returns the g_nmda_value corresponding with the membrane potential (V_m).
		*
		* \param V_m membrane potential.
		*
		* \return g_nmda_value corresponding with the membrane potential.
		*/
		virtual float Get_g_nmda_inf(float V_m) = 0;

		/*!
		* \brief It initializes the CurrentSynapsis object.
		*
		* It initializes the CurrentSynapsis object.
		*
		* \param N_neurons number of neurons.
		*/
		virtual void InitializeCurrentSynapsis(int N_neurons) = 0;


	public:

        /*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new neuron model.
		 *
		 * \param new_time_scale Variable that indicate which time scale implement this neuron model.
		 */
		TimeDrivenNeuronModel(TimeScale new_time_scale);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenNeuronModel();


		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
		 *
		 */
		virtual VectorNeuronState * InitializeState() = 0;


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
		virtual InternalSpike * ProcessInputSpike(Interconnection * inter, double time) = 0;

		/*!
		* \brief It processes a propagated current (input current in the cell).
		*
		* It processes a propagated current (input current in the cell).
		*
		* \param inter the interconection which propagate the spike
		* \param target the neuron which receives the spike
		* \param Current input current.
		*/
		virtual void ProcessInputCurrent(Interconnection * inter, Neuron * target, float current) = 0;


		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the vector. if index=-1, updating all cell.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(int index, double CurrentTime) = 0;


		/*!
		 * \brief It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * \return The simulation method of the neuron model.
		 */
		enum NeuronModelSimulationMethod GetModelSimulationMethod();

		/*!
		 * \brief It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * \return The neuron model type
		 */
		enum NeuronModelType GetModelType();

		/*!
		 * \brief It gets the neuron output activity type (spikes or currents).
		 *
		 * It gets the neuron output activity type (spikes or currents).
		 *
		 * \return The neuron output activity type (spikes or currents).
		 */
		virtual enum NeuronModelOutputActivityType GetModelOutputActivityType() = 0;

		/*!
		 * \brief It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * \return The neuron input activity types (spikes and/or currents or none).
		 */
		virtual enum NeuronModelInputActivityType GetModelInputActivityType() = 0;


		/*!
		 * \brief It prints the neuron model info.
		 *
		 * It prints the current neuron model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;

		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 * \param OpenMPQueueIndex openmp index
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex)=0;


		/*!
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return If the neuron model supports this connection type
		 */
		virtual bool CheckSynapseType(Interconnection * connection) = 0;


		/*!
		 * \brief It Checks if the integrations has work properly.
		 *
		 * It Checks if the integrations has worked properly.
		 *
		 * \param current time
		 */
		void CheckValidIntegration(double CurrentTime);

		/*!
		* \brief It Checks if the integrations has work properly.
		*
		* It Checks if the integrations has worked properly.
		*
		* \param current time
		* \param valid integration acumulated value of all the membranen potential computed on the integration method.
		*/
		void CheckValidIntegration(double CurrentTime, float valid_integration);


		/*!
		 * \brief It initializes an auxiliar array for time dependente variables.
		 *
		 * It initializes an auxiliar array for time dependente variables.
		 *
		 * \param N_conductances .
		 * \param N_elapsed_times .
		 */
		void Initialize_conductance_exp_values(int N_conductances, int N_elapsed_times);

		/*!
		 * \brief It calculates the conductace exponential value for an elapsed time.
		 *
		 * It calculates the conductace exponential value for an elapsed time.
		 *
		 * \param index elapsed time index .
		 * \param elapses_time elapsed time.
		 */
		virtual void Calculate_conductance_exp_values(int index, float elapsed_time)=0;

		/*!
		 * \brief It sets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * It sets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * \param elapses_time_index elapsed time index.
		 * \param conductance_index conductance index.
		 * \param value.
		 */
		void Set_conductance_exp_values(int elapsed_time_index, int conductance_index, float value);

		/*!
		 * \brief It gets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * It gets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * \param elapses_time_index elapsed time index.
		 * \param conductance_index conductance index.
		 *
		 * \return A conductance exponential values.
		 */
		float Get_conductance_exponential_values(int elapsed_time_index, int conductance_index);

		/*!
		 * \brief It gets the conductace exponential value for an elapsed time.
		 *
		 * It gets the conductace exponential value for an elapsed time .
		 *
		 * \param elapses_time_index elapsed time index.
		 *
		 * \return A pointer to a set of conductance exponential values.
		 */
		float * Get_conductance_exponential_values(int elapsed_time_index);

		/*!
		* \brief It calculates the number of electrical coupling synapses.
		*
		* It calculates the number for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		virtual void CalculateElectricalCouplingSynapseNumber(Interconnection * inter) = 0;

		/*!
		* \brief It allocate memory for electrical coupling synapse dependencies.
		*
		* It allocate memory for electrical coupling synapse dependencies.
		*/
		virtual void InitializeElectricalCouplingSynapseDependencies() = 0;

		/*!
		* \brief It calculates the dependencies for electrical coupling synapses.
		*
		* It calculates the dependencies for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		virtual void CalculateElectricalCouplingSynapseDependencies(Interconnection * inter) = 0;


		/*!
		* \brief It obtains the number of input conductances.
		*
		* It obtains the number of input conductances.
		*/
		int GetNConductances() const {
			return N_conductances;
		};


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
		virtual void GetBifixedStepParameters(float & startVoltageThreshold, float & endVoltageThreshold, float & timeAfterEndVoltageThreshold)=0;

		/*!
		* \brief It initialieses the structure required to implement external input current synapsis.
		*
		* It initialieses the structure required to implement external input current synapsis.
		*/
		void InitializeInputCurrentSynapseStructure();

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
		virtual std::map<std::string, boost::any> GetSpecificNeuronParameters(int index) const noexcept(false) = 0;

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
		* \brief Pure virtual function that must be implemented on the derived neuron models creating the integration method
		*
		* It creates the integration methods using the parameter map.
		*
		* \param param_map The dictionary with the integration method parameters.
		*
		* \throw EDLUTException If it happens a mistake with the parameters in the dictionary.
		*/
		virtual IntegrationMethod * CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false) = 0;

		/*!
		 * \brief It returns the default parameters of the neuron model.
		 *
		 * It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		 * set for this neuron model. This function has been defined as template because it need the neuron model type
		 * of the derived classes.
		 *
		 * \returns A dictionary with the neuron model default parameters.
		 */
		template <class Neuron_Model>
		static std::map<std::string,boost::any> GetDefaultParameters() {
			// Return a dictionary with the parameters
			std::map<std::string,boost::any> newMap = TimeDrivenModel::GetDefaultParameters();

			ModelDescription intMethod;
			intMethod.param_map = IntegrationMethodFactory<Neuron_Model>::GetDefaultParameters("Euler");
			intMethod.model_name = boost::any_cast<std::string>(intMethod.param_map["name"]);
			newMap["int_meth"] = boost::any(intMethod);
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
		template <class Neuron_Model>
		static ModelDescription ParseIntegrationMethod(FILE * fh, long & Currentline) noexcept(false){
			char ident_type[MAXIDSIZE + 1];
			skip_comments(fh, Currentline);
			if (fscanf(fh, "%64s", ident_type) != 1) {
				throw EDLUTException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_READ, REPAIR_INTEGRATION_METHOD_READ);
			}

			ModelDescription intMethodDescription = IntegrationMethodFactory<Neuron_Model>::ParseIntegrationMethod(ident_type, fh, Currentline);
			return intMethodDescription;
		}


		/*!
		 * \brief Comparison operator between neuron models.
		 *
		 * It compares two neuron models.
		 *
		 * \return True if the neuron models are of the same type and with the same parameters.
		 */
		virtual bool compare(const NeuronModel * rhs) const{
			if (!TimeDrivenModel::compare(rhs)){
				return false;
			}
			const TimeDrivenNeuronModel * e = dynamic_cast<const TimeDrivenNeuronModel *> (rhs);
			if (e == 0) return false;

			return this->N_conductances==e->N_conductances &&
				this->integration_method->compare(e->integration_method);
		};


};

#endif /* TIMEDRIVENNEURONMODEL_H_ */
