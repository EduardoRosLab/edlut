/***************************************************************************
 *                           NeuronPropagationDelayStructure.h             *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros					   *
 * email                : fnaveros@ugr.es		                           *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef NEURONPROPAGATIONDELAYSTRUCTURE_H_
#define NEURONPROPAGATIONDELAYSTRUCTURE_H_


class Neuron;
class Interconnection;
class NeuronModelPropagationDelayStructure;

/*!
 * \file NeuronPropagationDelayStructure.h
 *
 * \author Francisco Naveros
 * \date February 2015
 *
 * This file declares a class which define the propagation delay structure for each individual neuron. This
 * structure will be used to groupe the propagation event of different neurons in one single event.
 */
class NeuronPropagationDelayStructure{
	public:

		int * NDifferentDelays;
		int ** NSynapsesWithEqualDelay;
		double ** SynapseDelay;
		Interconnection*** OutputConnectionsWithEquealDealy;
		int ** IndexSynapseDelay;


   		/*!
		 * \brief Default constructor without parameters.
		 */
   		NeuronPropagationDelayStructure(Neuron * neuron);


		/*!
		 * \brief Default destructor.
		 */
   		~NeuronPropagationDelayStructure();


		void CalculateOutputDelayIndex(NeuronModelPropagationDelayStructure * PropagationStructure);
   		
 

};
  
#endif /*NEURONPROPAGATIONDELAYSTRUCTURE_H_*/
