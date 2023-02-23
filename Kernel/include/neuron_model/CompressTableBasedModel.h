/***************************************************************************
 *                           CompressTableBasedModel.h                     *
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

#ifndef COMPRESSTABLEBASEDMODEL_H_
#define COMPRESSTABLEBASEDMODEL_H_

/*!
 * \file CompressTableBasedModel.h
 *
 * \author Francisco Naveros
 * \date July 2015
 *
 * This file declares a class which implements a neuron model based in
 * look-up tables. This one works as the TableBasedModel class. The main
 * difference it is that this class try to compress some tables in only
 * one.
 */

#include "EventDrivenNeuronModel.h"

#include "../spike/EDLUTFileException.h"

class CompressNeuronModelTable;
class Interconnection;

/*!
* \brief Maximum number of state variables that the TableBasedModel can manage inside a neuron model.
*/
#define TABLE_MAX_VARIABLES 10

/*!
 * \class CompressTableBasedModel
 *
 * \brief Spiking neuron model based in look-up tables
 *
 * This class implements the behavior of event-driven spiking neuron models using
 * precalculated look-up tables to "predict" the neuron behavior. This "Compress"
 * version can merge several look-up tables with the same indexes in just one, minimizing
 * the look-up time (ideal for complex neuron models such as HH with several state
 * variables and look-up tables).
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 *
 * \author Francisco Naveros
 * \date July 2015
 */
class CompressTableBasedModel: public EventDrivenNeuronModel {
	protected:
		/*!
		 * \brief Number of state variables (no include time).
		 */
		unsigned int NumStateVar;

		/*!
		 * \brief Number of time dependent state variables.
		 */
		unsigned int NumTimeDependentStateVar;

		/*!
		 * \brief Number of synaptic variables.
		 */
		unsigned int NumSynapticVar;

		/*!
		 * \brief Index of synaptic variables.
		 */
		unsigned int * SynapticVar;

		/*!
		 * \brief Original order of state variables.
		 */
		unsigned int * StateVarOrderOriginalIndex;

		/*!
		 * \brief New order of state variables in compressed tables (some state variables will be stored in only one table).
		 */
		unsigned int * StateVarOrderIndex;

		/*!
		 * \brief New sub index inside of the state variables in compressed tables.
		 */
		unsigned int * StateVarOrderSubIndex;



		unsigned int * TablesIndex;

		/*!
		 * \brief Table which calculates each state variable.
		 */
		CompressNeuronModelTable ** StateVarTable;

		/*!
		 * \brief Firing time table
		 */
		CompressNeuronModelTable * FiringTable;

		/*!
		 * \brief Index of the firing time table
		 */
		unsigned int FiringIndex;

		/*!
		 * \brief Sub index inside the compressed table
		 */
		unsigned int FiringSubIndex;

		/*!
		 * \brief End firing time table
		 */
		CompressNeuronModelTable * EndFiringTable;

		/*!
		 * \brief Index of the end firing time table
		 */
		unsigned int EndFiringIndex;

		/*!
		 * \brief Sub index inside the compressed table
		 */
		unsigned int EndFiringSubIndex;

		/*!
		 * \brief Number of original tables
		 */
		unsigned int NumTables;

		/*!
		 * \brief Number of final compressed tables
		 */
		unsigned int NumCompresedTables;

		/*!
		 * \brief Precalculated tables
		 */
		CompressNeuronModelTable * Tables;

		/*!
		 * \brief Vector where we temporary store initial values
		 */
		float * InitValues;

		/*!
		* \brief String where is stored the name of the configuration file where are stored the look-up table parameters ("file.cfg").
		*/
		string conf_filename;

		/*!
		* \brief String where is stored the name of the file where are stored the look-up tables ("file.dat").
		*/
		string tab_filename;

		/*!
		 * \brief Number of state variables that store each compressed table
		 */
		int * NumVariablesPerCompressedTable;

		/*!
		 * \brief It loads the neuron model description.
		 *
		 * It loads the neuron type description from the file .cfg.
		 *
		 * \param ConfigFile Name of the neuron description file (*.cfg).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel(string ConfigFile) noexcept(false);

		/*!
		 * \brief It loads the neuron model tables.
		 *
		 * It loads the neuron model tables from his .dat associated file.
		 *
		 * \pre The neuron model must be previously initialized or loaded
		 *
		 * \param TableFile Name of the table file (*.dat).
		 *
		 * \see LoadNeuronModel()
		 * \throw EDLUTException If something wrong has happened in the tables loads.
		 */
		virtual void LoadTables(string TableFile) noexcept(false);

		/*!
		 * \brief It returns the end of the refractory period.
		 *
		 * It returns the end of the refractory period.
		 *
		 * \param index index inside the VectorNeuronState of the neuron model.
		 * \param VectorNeuronState of the neuron model.
		 *
		 * \return The end of the refractory period. -1 if no spike is predicted.
		 */
		virtual double EndRefractoryPeriod(int index, VectorNeuronState * State);

		/*!
		 * \brief It updates the neuron state after the evolution of the time.
		 *
		 * It updates the neuron state after the evolution of the time.
		 *
		 * \param index index inside the VectorNeuronState of the neuron model.
		 * \param VectorNeuronState of the neuron model.
		 * \param CurrentTime Current simulation time.
		 */
		virtual void UpdateState(int index, VectorNeuronState * State, double CurrentTime);

		/*!
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *
		 * \param index index inside the VectorNeuronState of the neuron model.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		virtual void SynapsisEffect(int index, Interconnection * InputConnection);


		/*!
		 * \brief It returns the next spike time.
		 *
		 * It returns the next spike time.
		 *
		 * \param index index inside the VectorNeuronState of the neuron model.
		 * \param VectorNeuronState of the neuron model.
		 *
		 * \return The next firing spike time. -1 if no spike is predicted.
		 */
		virtual double NextFiringPrediction(int index, VectorNeuronState * State);

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new default neuron model object. The configuration parameters and look-up table will be loaded in other function.
		 *
		 */
		CompressTableBasedModel();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~CompressTableBasedModel();

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel() noexcept(false);

		/*!
		 * \brief It tries to compact the neuron tables.
		 *
		 * It tries to compact the neuron tables.
		 */
		virtual void CompactTables();

		/*!
		 * \brief It compares if two tables have the same index and interpolation method and can be compressed in only one table.
		 *
		 * It compares if two tables have the same index and interpolation method and can be compressed in only one table.
		 *
		 * \param table1 first table.
		 * \param table2 second table.
		 *
		 * \return Boolean value that return if both tables are equal.
		 */
		virtual bool CompareNeuronModelTableIndex(CompressNeuronModelTable * table1, CompressNeuronModelTable * table2);

		/*!
		 * \brief It creates the neuron state and initializes to defined values.
		 *
		 * It creates the neuron state and initializes to defined values.
		 *
		 * \return A new object with the neuron state.
		 */
		virtual VectorNeuronState * InitializeState();

		/*!
		 * \brief It generates the first spike (if any) in a cell.
		 *
		 * It generates the first spike (if any) in a cell.
		 *
		 * \param Cell The cell to check if activity is generated.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * GenerateInitialActivity(Neuron *  Cell);


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
		 * \brief It predicts if the neuron would generate a internalSpike aftern all the propagated spikes have arrived.
		 *
		 * It predicts if the neuron would generate a internalSpike after all the propagated spikes have arrived.
		 *
		 * \param target Neuron that must be updated.
		 * \param time time.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessActivityAndPredictSpike(Neuron * target, double time);


		/*!
		 * \brief It processes an internal spike and generates an end refractory period event.
		 *
		 * It processes an internal spike and generates an end refractory period event.
		 *
		 * \param OutputSpike The spike happened.
		 *
		 * \return A new end refractory period event.
		 */
		virtual EndRefractoryPeriodEvent * ProcessInternalSpike(InternalSpike * OutputSpike);

		/*!
		 * \brief It calculates if an internal spike must be generated at the end of the refractory period.
		 *
		 * It calculates if an internal spike must be generated at the end of the refractory period.
		 *
		 * \param time end of the refractory period.
		 * \param neuron source neuron.
		 *
		 * \return A new internal spike.
		 */
		virtual InternalSpike * GenerateNextSpike(double time, Neuron * neuron);

		/*!
		 * \brief Check if the spike must be discard.
		 *
		 * Check if the spike must be discard. A spike must be discard if there are discrepancies between
		 * the next predicted spike and the spike time.
		 *
		 * \param OutputSpike The spike happened.
		 *
		 * \return True if the spike must be discard. False in otherwise.
		 */
		virtual bool DiscardSpike(InternalSpike *  OutputSpike);


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
		 * \brief It prints the table based model info.
		 *
		 * It prints the current table based model characteristics.
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
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
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
		 * \return A newly created InputNeuronModel object.
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
            if (!EventDrivenNeuronModel::compare(rhs)){
                return false;
            }
			const CompressTableBasedModel * e = dynamic_cast<const CompressTableBasedModel *> (rhs);
            if (e == 0) return false;

			return this->NumStateVar == e->NumStateVar &&
				this->NumTimeDependentStateVar == e->NumTimeDependentStateVar &&
				this->NumSynapticVar == e->NumSynapticVar &&
				this->SynapticVar == e->SynapticVar &&
				this->StateVarOrderOriginalIndex == e->StateVarOrderOriginalIndex &&
				this->StateVarOrderIndex == e->StateVarOrderIndex &&
				this->StateVarOrderSubIndex == e->StateVarOrderSubIndex &&
				this->TablesIndex == e->TablesIndex &&
				this->StateVarTable == e->StateVarTable &&
				this->FiringTable == e->FiringTable &&
				this->FiringIndex == e->FiringIndex &&
				this->FiringSubIndex == e->FiringSubIndex &&
				this->EndFiringTable == e->EndFiringTable &&
				this->EndFiringIndex == e->EndFiringIndex &&
				this->EndFiringSubIndex == e->EndFiringSubIndex &&
				this->NumTables == e->NumTables &&
				this->NumCompresedTables == e->NumCompresedTables &&
				this->Tables == e->Tables &&
				this->InitValues == e->InitValues &&
				this->NumVariablesPerCompressedTable == e->NumVariablesPerCompressedTable;
        };
};

#endif /* COMPRESSTABLEBASEDMODEL_H_ */
