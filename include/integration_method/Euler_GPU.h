/***************************************************************************
 *                           Euler_GPU.h                                   *
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

#ifndef EULER_GPU_H_
#define EULER_GPU_H_

/*!
 * \file Euler_GPU.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement the Euler integration method in GPU (this class is stored
 * in CPU memory and controles the allocation and deleting of GPU auxiliar memory). All integration
 * methods in GPU are fixed step due to the parallel architecture of this one.
 */

#include "./IntegrationMethod_GPU.h"


/*!
 * \class Euler
 *
 * \brief Euler integration method in CPU for GPU.
 *
 * This class abstracts the initializacion in CPU of an Euler integration methods for GPU. This CPU class
 * controles the reservation and freeing of GPU auxiliar memory.
 *
 * \author Francisco Naveros
 * \date May 2013
 */


class Euler_GPU: public IntegrationMethod_GPU{
	public:

		/*!
		 * \brief This vector is used as auxiliar vector.
		*/
		float * AuxNeuronState;


		/*!
		 * \brief Constructor of the class with 3 parameter.
		 *
		 * It generates a new Euler object.
		 *
		 * \param N_neuronStateVariables number of state variables for each cell.
		 * \param N_differentialNeuronState number of state variables witch are calculate with a differential equation for each cell.
		 * \param N_timeDependentNeuronState number of state variables witch ara calculate with a time dependent equation for each cell.
		 */
		Euler_GPU(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~Euler_GPU();


		/*!
		 * \brief This method reserves all the necesary GPU memory (this memory could be reserved directly in the GPU, but this 
		 * suppose some restriction in the amount of memory witch can be reserved).
		 *
		 * This method reserves all the necesary GPU memory (this memory could be reserved directly in the GPU, but this 
		 * suppose some restriction in the amount of memory witch can be reserved).
		 *
		 * \param N_neurons Number of neurons.
		 * \param Total_N_thread Number of thread in GPU.
		 */
		void InitializeMemoryGPU(int N_neurons, int Total_N_thread);
};

#endif /* EULER_GPU_H_ */
