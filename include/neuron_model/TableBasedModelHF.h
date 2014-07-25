/***************************************************************************
 *                           TableBasedModelHF.h                           *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
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

#ifndef TABLEBASEDMODELHF_H_
#define TABLEBASEDMODELHF_H_

/*!
 * \file TableBasedModelHF.h
 *
 * \author Francisco Naveros
 * \date April 2014
 *
 * This file is a modification of the TableBasedModel. This file declares a class which 
 * implements a neuron model based in look-up tables. The main difference it is that when
 * a input spike arrive to this model, the neuron state variables are update, but instead of
 * make a predicction in that instant, an event of type TableBasedModelHFEvent is created
 * with a delay of "ToleranceTime" (normaly fixed to 1us). All the spike that arrive inside 
 * this microsecond are computed conjointly and only a predicction is made. 
 * IMPORTANT: This method is better than the traditional TableBasedModel when this one 
 * receives input activity in a synchronize way, because only one prediction is made. Conversely,
 * when the input activity does not arrive in a synchronize way, this method create an innecesary
 * the overhead due to the creation of the TableBasedModelHFEvent.
 * 
 */

#include "EventDrivenNeuronModel.h"

#include "../spike/EDLUTFileException.h"

class NeuronModelTable;
class Interconnection;
class TableBasedModelHFEvent;

/*!
 * \class TableBasedModel
 *
 * \brief Spiking neuron model based in look-up tables
 *
 * This class implements the behavior of a neuron in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This behavior is calculated based in precalculated look-up tables.
 *
 * \author Jesus Garrido
 * \date February 2010
 */
class TableBasedModelHF: public EventDrivenNeuronModel {
	protected:

		/*!
		 * \brief Time between the arrive of the first input spike and the execution of the
		 *  TableBasedModelHFEvent to predict if the neuron will spike or not.
		 */
		double ToleranceTime;

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
		 * \brief Order of state variables.
		 */
		unsigned int * StateVarOrder;

		/*!
		 * \brief Table which calculates each state variable.
		 */
		NeuronModelTable ** StateVarTable;

		/*!
		 * \brief Firing time table
		 */
		NeuronModelTable * FiringTable;

		/*!
		 * \brief End firing time table
		 */
		NeuronModelTable * EndFiringTable;

		/*!
		 * \brief Number of tables
		 */
		unsigned int NumTables;

		/*!
		 * \brief Precalculated tables
		 */
		NeuronModelTable * Tables;


		/*!
		 * \brief Vector where we temporary store initial values
		 */
		float * InitValues;

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
		virtual void LoadTables(string TableFile) throw (EDLUTException);

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
		 * It generates a new neuron model object loading the configuration of
		 * the model and the look-up tables.
		 *
		 * \param NeuronTypeID Neuron model type.
		 * \param NeuronModelID Neuron model description file.
		 */
		TableBasedModelHF(string NeuronTypeID, string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~TableBasedModelHF();

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel() throw (EDLUTFileException);

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
		 * \param InputSpike The spike happened.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessInputSpike(PropagatedSpike *  InputSpike);


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
		 * \brief It predicts if the neuron would generate a internalSpike after the ToleranceTime. 
		 *
		 * It predicts if the neuron would generate a internalSpike after the ToleranceTime. 
		 *
		 * \param target.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		InternalSpike * ProcessActivityAndPredictSpike(Neuron * target);

		/*!
		 * \brief It processes an internal spike (generated spike in the cell).
		 *
		 * It processes an internal spike (generated spike in the cell).
		 *
		 * \note This function doesn't generate the next propagated (output) spike. It must be externally done.
		 * \note Before generating next spike, you should check if this spike must be discard.
		 *
		 * \see DiscardSpike
		 *
		 * \param OutputSpike The spike happened.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * GenerateNextSpike(InternalSpike *  OutputSpike);

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
		 */
		virtual void InitializeStates(int N_neurons);

};

#endif /* TABLEBASEDMODELHF_H_ */
