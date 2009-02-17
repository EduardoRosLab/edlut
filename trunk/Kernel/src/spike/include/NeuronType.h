/***************************************************************************
 *                           NeuronType.h                                  *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef NEURONTYPE_H_
#define NEURONTYPE_H_

/*!
 * \file NeuronType.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a neuron model.
 */
#include <cstdlib>
#include <cstring>

#include "../../simulation/include/Configuration.h"
#include "./EDLUTFileException.h"
#include "./NeuronModelTable.h"

/*!
 * \class NeuronType
 *
 * \brief Neuron model
 *
 * This class abstract the behaviour of a neuron model of a spiking neural network.
 * It includes the name, the number of variables, the initial values, the neuron model tables...
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class NeuronType{
	private:
	
		/*!
		 * Neuron model name.
		 */	
		char ident[MAXIDSIZE+1];
   		
   		/*!
   		 * Number of state variables.
   		 */
   		unsigned int nstatevars;
   		
   		/*!
   		 * Number of time dependent state variables.
   		 */
   		unsigned int ntstatevars;
   		
   		/*!
   		 * Order of the state variables.
   		 */
   		int statevarorder[MAXSTATEVARS];
   		
   		/*!
   		 * Tables of the state variables.
   		 */
   		int statevartables[MAXSTATEVARS];
   		
   		/*!
   		 * Initial values.
   		 */
   		float initvalues[MAXSTATEVARS];
   		
   		/*!
   		 * Number of the firing table.
   		 */
   		int firingtable;
   		
   		/*!
   		 * Number of the firing end table.
   		 */
   		int firingendtable;
   		
   		/*!
   		 * Number of synaptic variables.
   		 */
   		unsigned int nsynapticvars;
   		
   		/*!
   		 * Synaptic variables.
   		 */
   		int synapticvars[MAXSTATEVARS];
   		
   		/*!
   		 * Neuron model tables.
   		 */
   		NeuronModelTable tables[MAXSTATEVARS*2];
   		
   		/*!
   		 * Number of neuron model tables.
   		 */
   		unsigned int ntables;
   		   		
   	public:
   	
   		/*!
   		 * \brief It gets the neuron model identificator.
   		 * 
   		 * It gets the neuron model name.
   		 * 
   		 * \return The neuron model identificator.
   		 */
   		char * GetId();
   		
   		/*!
   		 * \brief It gets the number of state variables.
   		 * 
   		 * It gets the number of state variables of the neuron model.
   		 * 
   		 * \return The number of state variables.
   		 */
   		int GetStateVarsNumber() const;
   		
   		/*!
   		 * \brief It gets the number of time dependent state variables.
   		 * 
   		 * It gets the number of time dependent state variables of the neuron model.
   		 * 
   		 * \return The number of time dependent state variables.
   		 */
   		int GetTimeDependentStateVarsNumber() const;
   		
   		/*!
   		 * \brief It gets the number of a state variable.
   		 * 
   		 * It gets the number of a state variable from the index.
   		 * 
   		 * \param index The index of the state variable to get.
   		 * 
   		 * \return The number of the indexth state variable.
   		 */
   		int GetStateVarAt(int index) const;
   		
   		/*!
   		 * \brief It gets the table of a concrete variable.
   		 * 
   		 * It gets the index of the table of the indexth variable.
   		 * 
   		 * \param index The index of the variable.
   		 * 
   		 * \return The number of the table with the state variable index.
   		 */
   		int GetStateVarTableAt(int index) const;
   		
   		/*!
   		 * \brief It gets the initial value of a variable.
   		 * 
   		 * It gets the initial value of a concrete state variable.
   		 * 
   		 * \param index The index of the variable to get the initial value.
   		 * 
   		 * \return The initial value of the indexth variable.
   		 */
   		float GetInitValueAt(int index) const;
   		
   		/*!
   		 * \brief It gets the index of the firing table.
   		 * 
   		 * It gets the index of the firing table.
   		 * 
   		 * \return The index of the firing table.
   		 */
   		int GetFiringTable() const;
   		
   		/*!
   		 * \brief It gets the index of the end firing table.
   		 * 
   		 * It gets the index of the end firing table.
   		 * 
   		 * \return The index of the end firing table.
   		 */
   		int GetFiringEndTable() const;
   		
   		/*!
   		 * \brief It gets the number of the synaptic variables.
   		 * 
   		 * It gets the number of the synaptic variables.
   		 * 
   		 * \return The number of synaptic variables.
   		 */
   		int GetSynapticVarsNumber() const;
   		
   		/*!
   		 * \brief It gets the number of a synaptic variable.
   		 * 
   		 * It gets the number of a synaptic variable from the index.
   		 * 
   		 * \param index The index of the synaptic variable to get.
   		 * 
   		 * \return The number of the indexth synaptic variable.
   		 */
   		int GetSynapticVarsAt(int index) const;
   		
   		/*!
   		 * \brief It gets a neuron model table.
   		 * 
   		 * It gets a neuron model table from the index.
   		 * 
   		 * \param index The index of the neuron model table to get.
   		 * 
   		 * \return The number of the indexth neuron model table.
   		 */
   		NeuronModelTable * GetTableAt(int index);
   		
   		/*!
   		 * \brief It gets the number of neuron model tables.
   		 * 
   		 * It gets the number of neuron model tables.
   		 * 
   		 * \return The number of neuron model tables.
   		 */
   		int GetTableNumber() const;
   		
   		/*!
   		 * \brief It clears the neuron type for loading a new model.
   		 * 
   		 * It clears the neuron type, his identificator, the state variables and the neuron model tables.
   		 */
   		void ClearNeuronType();
   		
   		/*!
   		 * \brief It loads the neuron model description.
   		 * 
   		 * It loads the neuron type description from the file .cfg.
   		 * 
   		 * \param neutype Name of the neuron type (without file extension).
   		 * 
   		 * \throw EDLUTFileException If something wrong has happened in the file load.
   		 */
   		void LoadNeuronType(char * neutype) throw (EDLUTFileException);
   		
   		/*!
   		 * \brief It loads the neuron model tables.
   		 * 
   		 * It loads the neuron model tables from his .dat associated file.
   		 * 
   		 * \pre The neuron type must be previously initialized or loaded
   		 * 
   		 * \see LoadNeuronType()
   		 * \throw EDLUTException If something wrong has happened in the tables loads.
   		 */
   		void LoadTables() throw (EDLUTException);
   		
   		//void NeutypeInfo();
   		
   		/*!
   		 * \brief It gets a value from the table.
   		 * 
   		 * It gets the value associated a the table ntab with the statevars nueral state.
   		 * 
   		 * \pre The tables must be previously loaded.
   		 * 
   		 * \param ntab Number of the table to get the value.
   		 * 
   		 * \param statevars State variables values of the neuron.
   		 */
   		float TableAccess(int ntab, float *statevars);
};

#endif /*NEURONTYPE_H_*/
