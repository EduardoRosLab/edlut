/***************************************************************************
 *                           EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE.cuh *
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

#ifndef EGIDIOGRANULECELL_TIMEDRIVEN_GPU_C_INTERFACE_H_
#define EGIDIOGRANULECELL_TIMEDRIVEN_GPU_C_INTERFACE_H_


/*!
 * \file EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE.cuh
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model for a cerebellar 
 * granule cell. This neuron model has 15 differential equations and 3 time dependent equations (conductances) and
 * one input current. This model is implemented in CPU to control a GPU class.
 */

#include "./TimeDrivenNeuronModel_GPU_C_INTERFACE.cuh"

#include <string>

using namespace std;

class InputSpike;
class VectorNeuronState;
class VectorNeuronState_GPU_C_INTERFACE;
class Interconnection;

class EgidioGranuleCell_TimeDriven_GPU2;

/*!
 * \class EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model with fifteen differentical equations,
 * three conductances and one input current. This model is implemented in CPU to control a GPU class.
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
class EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE : public TimeDrivenNeuronModel_GPU_C_INTERFACE {
	protected:

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

		// gating-kinetic correction for experiment temperature=20�C;
		const float Q10_20;
		// gating-kinetic correction for experiment temperature=22�C;
		const float Q10_22;
		// experiment temperature=30�C;
		const float Q10_30;
		// experiment temperature=6.3�C;
		const float Q10_6_3;

//		const float I_inj_abs;

		// Injected Current in absolute terms = -10pA;
		// Cell membrane area = pi*2*r*L = pi*9.76*9.76 = 299.26058um^2 = 299.26058*10^-8cm^2;
		// Injected current per area unit = -I_inj_abs/ 299.26058*10^-8cm^2 = I_inj;
//		const float I_inj;

		const float e_exc;
		const float e_inh;
		const float tau_exc;
		const float tau_inh;
		const float tau_nmda;
		const float v_thr;


		/*!
		 * \brief Neuron model in the GPU.
		*/
		EgidioGranuleCell_TimeDriven_GPU2 ** NeuronModel_GPU2;


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
		const int N_NeuronStateVariables=19;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		*/
		const int N_DifferentialNeuronState=15;

		/*!
		 * \brief Number of state variables which are calculate with a time dependent equation for each cell.
		*/
		const int N_TimeDependentNeuronState=4;
			//Index of each time dependent neuron state variable
			const int EXC_index = N_DifferentialNeuronState;
			const int INH_index = N_DifferentialNeuronState + 1;
			const int NMDA_index = N_DifferentialNeuronState + 2;
			const int EXT_I_index = N_DifferentialNeuronState + 3;

		
		/*!
		 * \brief Boolean variable setting in runtime if the neuron model receive each one of the supported input synpase types (N_TimeDependentNeuronState) 
		 */
		bool EXC, INH, NMDA, EXT_I;

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 */
		EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE();

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
		 * \brief It initialice a neuron model in GPU.
		 *
		 * It initialice a neuron model in GPU.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 */
		virtual void InitializeClassGPU2(int N_neurons);

		/*!
		 * \brief It delete a neuron model in GPU.
		 *
		 * It delete a neuron model in GPU.
		 */
		virtual void DeleteClassGPU2();

		/*!
		 * \brief It create a object of type VectorNeuronState_GPU2 in GPU.
		 *
		 * It create a object of type VectorNeuronState_GPU2 in GPU.
		 */
		virtual void InitializeVectorNeuronState_GPU2();

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
		* \brief It returns the neuron model parameters.
		*
		* It returns the neuron model parameters.
		*
		* \returns A dictionary with the neuron model parameters
		*/
		virtual std::map<std::string, boost::any> GetParameters() const;

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
		virtual IntegrationMethod_GPU_C_INTERFACE * CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false);

		/*!
		* \brief It returns the default parameters of the neuron model.
		*
		* It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		* set for this neuron model.
		*
		* \returns A dictionary with the neuron model default parameters.
		*/
		static std::map<std::string, boost::any> GetDefaultParameters();

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
			if (!TimeDrivenNeuronModel_GPU_C_INTERFACE::compare(rhs)){
				return false;
			}
			const EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE * e = dynamic_cast<const EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE *> (rhs);
			if (e == 0) return false;

			return true;
		};

};
#endif /* EGIDIOGRANULECELL_TIMEDRIVEN_GPU_H_ */
