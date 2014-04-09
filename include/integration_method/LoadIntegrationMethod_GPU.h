/***************************************************************************
 *                           LoadIntegrationMethod_GPU.h                   *
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

#ifndef LOADINTEGRATIONMETHOD_GPU_H_
#define LOADINTEGRATIONMETHOD_GPU_H_

/*!
 * \file LoadIntegrationMethod_GPU.h
 *
 * \author Francisco Naveros
 * \date November 2013
 *
 * This file declares a class which load all integration methods in CPU for GPU.
 */

#include <string>
using namespace std;

#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"

#include "./IntegrationMethod_GPU.h"
#include "./Euler_GPU.h"
#include "./RK2_GPU.h"
#include "./RK4_GPU.h"
#include "./BDFn_GPU.h"



#include "../../include/simulation/Utils.h"
#include "../../include/simulation/Configuration.h"




/*!
 * \class LoadIntegrationMethod_GPU
 *
 * \brief Load Integration methods in CPU for GPU
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class LoadIntegrationMethod_GPU {
	protected:

	public:

		static IntegrationMethod_GPU * loadIntegrationMethod_GPU(FILE *fh, long * Currentline, int N_NeuronStateVariables, int N_DifferentialNeuronState, int N_TimeDependentNeuronState)throw (EDLUTFileException){
			IntegrationMethod_GPU * Method;
			char ident_type[MAXIDSIZE+1];

			skip_comments(fh,*Currentline);
			if(fscanf(fh,"%s",ident_type)==1){
				skip_comments(fh,*Currentline);

				//DEFINE HERE NEW INTEGRATION METHOD
				if(strncmp(ident_type,"Euler",5)==0){
					Method=(Euler_GPU *) new Euler_GPU(N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"RK2",3)==0){
					Method=(RK2_GPU *) new RK2_GPU(N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"RK4",3)==0){
					Method=(RK4_GPU *) new RK4_GPU(N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"BDF",3)==0 && atoi(&ident_type[3])>0 && atoi(&ident_type[3])<7){
					Method=(BDFn_GPU *) new BDFn_GPU(N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState, atoi(&ident_type[3]), ident_type);
				}else{
////NEW CODE------------------------------------------------------------------------------
					throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
				}

			}else{
//NEW CODE------------------------------------------------------------------------------
				throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
			}
			return Method;
		}

		

};




#endif /* LOADINTEGRATIONMETHOD_H_ */
