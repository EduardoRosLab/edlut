/***************************************************************************
 *                           IntegrationMethodFactory_GPU_C_INTERFACE.cpp  *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Jesus Garrido                        *
 * email                : jesusgarrido@ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "integration_method/IntegrationMethodFactory_GPU_C_INTERFACE.cuh"
//GPU

//INTEGRATION METHODS
#include "integration_method/Euler_GPU_C_INTERFACE.cuh"
#include "integration_method/RK2_GPU_C_INTERFACE.cuh"
#include "integration_method/RK4_GPU_C_INTERFACE.cuh"
#include "integration_method/BDFn_GPU_C_INTERFACE.cuh"
#include "integration_method/Bifixed_Euler_GPU_C_INTERFACE.cuh"
#include "integration_method/Bifixed_RK2_GPU_C_INTERFACE.cuh"
#include "integration_method/Bifixed_RK4_GPU_C_INTERFACE.cuh"
//#include "integration_method/Bifixed_BDFn_GPU_C_INTERFACE.cuh"   //It doesn't work properly

//NEURONS MODELS
#include "neuron_model/AdExTimeDrivenModel_GPU_C_INTERFACE.cuh"
#include "neuron_model/EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE.cuh"
#include "neuron_model/HHTimeDrivenModel_GPU_C_INTERFACE.cuh"
#include "neuron_model/IzhikevichTimeDrivenModel_GPU_C_INTERFACE.cuh"
#include "neuron_model/LIFTimeDrivenModel_GPU_C_INTERFACE.cuh"
#include "neuron_model/LIFTimeDrivenModel_IS_GPU_C_INTERFACE.cuh"
#include "neuron_model/TimeDrivenPurkinjeCell_GPU_C_INTERFACE.cuh"


template class IntegrationMethodFactory_GPU_C_INTERFACE <AdExTimeDrivenModel_GPU_C_INTERFACE>;
template class IntegrationMethodFactory_GPU_C_INTERFACE <EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE>;
template class IntegrationMethodFactory_GPU_C_INTERFACE <HHTimeDrivenModel_GPU_C_INTERFACE>;
template class IntegrationMethodFactory_GPU_C_INTERFACE <IzhikevichTimeDrivenModel_GPU_C_INTERFACE>;
template class IntegrationMethodFactory_GPU_C_INTERFACE <LIFTimeDrivenModel_GPU_C_INTERFACE>;
template class IntegrationMethodFactory_GPU_C_INTERFACE <LIFTimeDrivenModel_IS_GPU_C_INTERFACE>;
template class IntegrationMethodFactory_GPU_C_INTERFACE <TimeDrivenPurkinjeCell_GPU_C_INTERFACE>;


template <class Neuron_Model>
std::map<std::string, class IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU > IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU;


template<class Neuron_Model>
void IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::InitializeIntegrationMethodFactory_GPU_C_INTERFACE(){
	if (IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU.empty()){
		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU[Euler_GPU_C_INTERFACE<Neuron_Model>::GetName()] =
			IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU(
			Euler_GPU_C_INTERFACE<Neuron_Model>::ParseIntegrationMethod,
			Euler_GPU_C_INTERFACE<Neuron_Model>::CreateIntegrationMethod,
			Euler_GPU_C_INTERFACE<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU[RK2_GPU_C_INTERFACE<Neuron_Model>::GetName()] =
			IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU(
			RK2_GPU_C_INTERFACE<Neuron_Model>::ParseIntegrationMethod,
			RK2_GPU_C_INTERFACE<Neuron_Model>::CreateIntegrationMethod,
			RK2_GPU_C_INTERFACE<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU[RK4_GPU_C_INTERFACE<Neuron_Model>::GetName()] =
			IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU(
			RK4_GPU_C_INTERFACE<Neuron_Model>::ParseIntegrationMethod,
			RK4_GPU_C_INTERFACE<Neuron_Model>::CreateIntegrationMethod,
			RK4_GPU_C_INTERFACE<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU[BDFn_GPU_C_INTERFACE<Neuron_Model>::GetName()] =
			IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU(
			BDFn_GPU_C_INTERFACE<Neuron_Model>::ParseIntegrationMethod,
			BDFn_GPU_C_INTERFACE<Neuron_Model>::CreateIntegrationMethod,
			BDFn_GPU_C_INTERFACE<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU[Bifixed_Euler_GPU_C_INTERFACE<Neuron_Model>::GetName()] =
			IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU(
			Bifixed_Euler_GPU_C_INTERFACE<Neuron_Model>::ParseIntegrationMethod,
			Bifixed_Euler_GPU_C_INTERFACE<Neuron_Model>::CreateIntegrationMethod,
			Bifixed_Euler_GPU_C_INTERFACE<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU[Bifixed_RK2_GPU_C_INTERFACE<Neuron_Model>::GetName()] =
			IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU(
			Bifixed_RK2_GPU_C_INTERFACE<Neuron_Model>::ParseIntegrationMethod,
			Bifixed_RK2_GPU_C_INTERFACE<Neuron_Model>::CreateIntegrationMethod,
			Bifixed_RK2_GPU_C_INTERFACE<Neuron_Model>::GetDefaultParameters);
		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU[Bifixed_RK4_GPU_C_INTERFACE<Neuron_Model>::GetName()] =
			IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU(
			Bifixed_RK4_GPU_C_INTERFACE<Neuron_Model>::ParseIntegrationMethod,
			Bifixed_RK4_GPU_C_INTERFACE<Neuron_Model>::CreateIntegrationMethod,
			Bifixed_RK4_GPU_C_INTERFACE<Neuron_Model>::GetDefaultParameters);
//		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU[Bifixed_BDFn_GPU_C_INTERFACE<Neuron_Model>::GetName()] =
//			IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU(
//			Bifixed_BDFn_GPU_C_INTERFACE<Neuron_Model>::ParseIntegrationMethod,
//			Bifixed_BDFn_GPU_C_INTERFACE<Neuron_Model>::CreateIntegrationMethod,
//			Bifixed_BDFn_GPU_C_INTERFACE<Neuron_Model>::GetDefaultParameters);
	}
}


template<class Neuron_Model>
IntegrationMethod_GPU_C_INTERFACE * IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::CreateIntegrationMethod_GPU(ModelDescription imethodDescription, Neuron_Model * nmodel){
	IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::InitializeIntegrationMethodFactory_GPU_C_INTERFACE();
	// Find the rule description name in the learning rule map
typename
	std::map<std::string, IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU >::const_iterator it =
		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU.find(imethodDescription.model_name);
	if (it == IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU.end()){
		throw EDLUTException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_TYPE,
			REPAIR_INTEGRATION_METHOD_TYPE);
	}
	IntegrationMethod_GPU_C_INTERFACE * imethod = it->second.createFunc(imethodDescription, nmodel);
	return imethod;
}



template<class Neuron_Model>
ModelDescription IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::ParseIntegrationMethod_GPU(std::string ident, FILE * fh, long & Currentline) {
	IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::InitializeIntegrationMethodFactory_GPU_C_INTERFACE();
	// Find the rule description name in the learning rule map
typename
	std::map<std::string, IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU >::const_iterator it =
		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU.find(ident);
	if (it == IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU.end()) {
		throw EDLUTException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_TYPE,
			REPAIR_INTEGRATION_METHOD_TYPE);
	}
	return it->second.parseFunc(fh, Currentline);
}



template<class Neuron_Model>
std::map<std::string, boost::any> IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::GetDefaultParameters_GPU(std::string ident){
	IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::InitializeIntegrationMethodFactory_GPU_C_INTERFACE();
	// Find the rule description name in the learning rule map
typename
	std::map<std::string, IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodClass_GPU >::const_iterator it =
		IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU.find(ident);
	if (it == IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::IntegrationMethodMap_GPU.end()) {
		throw EDLUTException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_TYPE,
			REPAIR_INTEGRATION_METHOD_TYPE);
	}
	return it->second.getDefaultParamFunc();
}
