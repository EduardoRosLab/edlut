/***************************************************************************
 *                           NeuronModelPropagationDelayStructure.h        *
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

#ifndef NEURONMODELPROPAGATIONDELAYSTRUCTURE_H_
#define NEURONMODELPROPAGATIONDELAYSTRUCTURE_H_


class Neuron;
class Interconnection;

/*!
 * \file NeuronmodelPropagationDelayStructure.h
 *
 * \author Francisco Naveros
 * \date February 2015
 *
 * This file declares a class which define the propagation delay structure for each individual neuron. This
 * structure will be used to groupe the propagation event of different neurons in one single event.
 */
class NeuronModelPropagationDelayStructure{
	double ** delays;
	int ** eventSize;
	int *AllocatedSize;
	int *size;

	public:

   		/*!
		 * \brief Default constructor without parameters.
		 */
   		NeuronModelPropagationDelayStructure();


		/*!
		 * \brief Default destructor.
		 */
   		~NeuronModelPropagationDelayStructure();


		void IncludeNewDelay(int queueIndex, double newDelay);

		int GetAllocatedSize(int queueIndex);

		int GetSize(int queueIndex);

		double GetDelayAt(int queueIndex, int index);

		double * GetDelays(int queueIndex);

		int GetEventSize(int queueIndex, int index);

		void IncrementEventSize(int queueIndex, int index);

	private:

		void FillEventSize(int queueIndex);

};
  
#endif /*NEURONMODELPROPAGATIONDELAYSTRUCTURE_H_*/
