/***************************************************************************
 *                           LoadIntegrationMethod.h                       *
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

#ifndef LOADINTEGRATIONMETHOD_H_
#define LOADINTEGRATIONMETHOD_H_

/*!
 * \file LoadIntegrationMethod.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which load all integration methods in a CPU.
 */

#include <string.h>
#include <cstdlib>
using namespace std;

#include "./IntegrationMethod.h"
#include "./Euler.h"
#include "./RK2.h"
#include "./RK4.h"
#include "./RK45.h"
#include "./RK45ad.h"
#include "./BDF1ad.h"
#include "./BDFn.h"

#include "./FixedStepSRM.h"
#include "./VariableStepSRM.h"


#include "../../include/simulation/Utils.h"
#include "../../include/simulation/Configuration.h"

class NeuronModel;


/*!
 * \class LoadIntegrationMethod
 *
 * \brief Load Integration methods in CPU
 *
 * \author Francisco Naveros
 * \date May 2012
 */
class LoadIntegrationMethod {
	protected:

	public:

		static IntegrationMethod * loadIntegrationMethod(TimeDrivenNeuronModel* model, FILE *fh, long * Currentline, int N_NeuronStateVariables, int N_DifferentialNeuronState, int N_TimeDependentNeuronState)throw (EDLUTFileException){
			IntegrationMethod * Method;
			char ident_type[MAXIDSIZE+1];

			//We load the integration method type.
			skip_comments(fh,*Currentline);
			if(fscanf(fh,"%s",ident_type)==1){
				skip_comments(fh,*Currentline);
				//DEFINE HERE NEW INTEGRATION METHOD
				if(strncmp(ident_type,"Euler",5)==0){
					Method=(Euler *) new Euler(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"RK2",3)==0){
					Method=(RK2 *) new RK2(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"RK45ad",6)==0){
					Method=(RK45ad *) new RK45ad(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"RK45",4)==0){
					Method=(RK45 *) new RK45(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"RK4",3)==0){
					Method=(RK4 *) new RK4(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"BDF1ad",6)==0 ){
					Method=(BDF1ad *) new BDF1ad(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"BDF",3)==0 && atoi(&ident_type[3])>0 && atoi(&ident_type[3])<7){
					Method=(BDFn *) new BDFn(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState,atoi(&ident_type[3]));
				}else if(strncmp(ident_type,"FixedStepSRM",12)==0){
					Method=(FixedStepSRM *) new FixedStepSRM();
				}else if(strncmp(ident_type,"VariableStepSRM",15)==0){
					Method=(VariableStepSRM *) new VariableStepSRM();
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


			//We load the integration method parameter.
			Method->loadParameter(fh,Currentline);
			

			return Method;
		}
};




#endif /* LOADINTEGRATIONMETHOD_H_ */
