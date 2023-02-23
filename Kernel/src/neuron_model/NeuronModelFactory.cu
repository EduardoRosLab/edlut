/***************************************************************************
 *                           NeuronModelFactory.cpp                        *
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
#include "neuron_model/NeuronModelFactory.h"

//EVENT DRIVEN INPUT DEVICES IN CPU
#include <neuron_model/InputCurrentNeuronModel.h>
#include <neuron_model/InputSpikeNeuronModel.h>

//TIME DRIVEN INPUT DEVICES IN CPU
#include <neuron_model/PoissonGeneratorDeviceVector.h>
#include <neuron_model/SinCurrentDeviceVector.h>


//TIME DRIVEN NEURON MODELS IN CPU
#include <neuron_model/AdExTimeDrivenModel.h>
#include <neuron_model/AdExTimeDrivenModelVector.h>
#include <neuron_model/EgidioGranuleCell_TimeDriven.h>
#include <neuron_model/HHTimeDrivenModel.h>
#include <neuron_model/IzhikevichTimeDrivenModel.h>
#include <neuron_model/LIFTimeDrivenModel.h>
#include <neuron_model/LIFTimeDrivenModel_IS.h>
#include <neuron_model/TimeDrivenInferiorOliveCell.h>
#include <neuron_model/TimeDrivenPurkinjeCell.h>
#include <neuron_model/TimeDrivenPurkinjeCell_IP.h>

//EVENT DRIVEN NEURON MODELS IN CPU
#include <neuron_model/CompressSynchronousTableBasedModel.h>
#include <neuron_model/CompressTableBasedModel.h>
#include <neuron_model/SynchronousTableBasedModel.h>
#include <neuron_model/TableBasedModel.h>


#ifdef USE_CUDA
	//TIME DRIVEN NEURON MODELS IN GPU
	#include <neuron_model/AdExTimeDrivenModel_GPU_C_INTERFACE.cuh>
	#include <neuron_model/EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE.cuh>
	#include <neuron_model/HHTimeDrivenModel_GPU_C_INTERFACE.cuh>
	#include <neuron_model/IzhikevichTimeDrivenModel_GPU_C_INTERFACE.cuh>
	#include <neuron_model/LIFTimeDrivenModel_GPU_C_INTERFACE.cuh>
	#include <neuron_model/LIFTimeDrivenModel_IS_GPU_C_INTERFACE.cuh>
	#include <neuron_model/TimeDrivenPurkinjeCell_GPU_C_INTERFACE.cuh>
#endif //USE_CUDA

#include "../../include/spike/EDLUTException.h"



std::map<std::string, NeuronModelFactory::NeuronModelClass > NeuronModelFactory::NeuronModelMap;


std::vector<std::string> NeuronModelFactory::GetAvailableNeuronModels(){
	std::vector<std::string> availableNeuronModels;
	availableNeuronModels.push_back(InputCurrentNeuronModel::GetName());
	availableNeuronModels.push_back(InputSpikeNeuronModel::GetName());
	availableNeuronModels.push_back(PoissonGeneratorDeviceVector::GetName());
	availableNeuronModels.push_back(SinCurrentDeviceVector::GetName());
	availableNeuronModels.push_back(AdExTimeDrivenModel::GetName());
	availableNeuronModels.push_back(AdExTimeDrivenModelVector::GetName());
	availableNeuronModels.push_back(EgidioGranuleCell_TimeDriven::GetName());
	availableNeuronModels.push_back(HHTimeDrivenModel::GetName());
	availableNeuronModels.push_back(IzhikevichTimeDrivenModel::GetName());
	availableNeuronModels.push_back(LIFTimeDrivenModel::GetName());
	availableNeuronModels.push_back(LIFTimeDrivenModel_IS::GetName());
	availableNeuronModels.push_back(TimeDrivenInferiorOliveCell::GetName());
	availableNeuronModels.push_back(TimeDrivenPurkinjeCell::GetName());
	availableNeuronModels.push_back(TimeDrivenPurkinjeCell_IP::GetName());
	availableNeuronModels.push_back(CompressSynchronousTableBasedModel::GetName());
	availableNeuronModels.push_back(CompressTableBasedModel::GetName());
	availableNeuronModels.push_back(SynchronousTableBasedModel::GetName());
	availableNeuronModels.push_back(TableBasedModel::GetName());
#ifdef USE_CUDA
	availableNeuronModels.push_back(AdExTimeDrivenModel_GPU_C_INTERFACE::GetName());
	availableNeuronModels.push_back(EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetName());
	availableNeuronModels.push_back(HHTimeDrivenModel_GPU_C_INTERFACE::GetName());
	availableNeuronModels.push_back(IzhikevichTimeDrivenModel_GPU_C_INTERFACE::GetName());
	availableNeuronModels.push_back(LIFTimeDrivenModel_GPU_C_INTERFACE::GetName());
	availableNeuronModels.push_back(LIFTimeDrivenModel_IS_GPU_C_INTERFACE::GetName());
	availableNeuronModels.push_back(TimeDrivenPurkinjeCell_GPU_C_INTERFACE::GetName());
#endif //USE_CUDA

	return availableNeuronModels;
}

void NeuronModelFactory::PrintAvailableNeuronModels(){
	std::vector<std::string> availableNeuronModels = GetAvailableNeuronModels();
	cout << "- Available Neuron Models in EDLUT:" << endl;
	for (std::vector<std::string>::const_iterator it = availableNeuronModels.begin(); it != availableNeuronModels.end(); ++it){
		std::cout <<"\t- "<< *it << std::endl;
	}
}


void NeuronModelFactory::InitializeNeuronModelFactory(){
	if (NeuronModelFactory::NeuronModelMap.empty()){
		NeuronModelFactory::NeuronModelMap[InputCurrentNeuronModel::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				InputCurrentNeuronModel::ParseNeuronModel,
				InputCurrentNeuronModel::CreateNeuronModel,
				InputCurrentNeuronModel::GetDefaultParameters,
				InputCurrentNeuronModel::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[InputSpikeNeuronModel::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				InputSpikeNeuronModel::ParseNeuronModel,
				InputSpikeNeuronModel::CreateNeuronModel,
				InputSpikeNeuronModel::GetDefaultParameters,
				InputSpikeNeuronModel::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[PoissonGeneratorDeviceVector::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				PoissonGeneratorDeviceVector::ParseNeuronModel,
				PoissonGeneratorDeviceVector::CreateNeuronModel,
				PoissonGeneratorDeviceVector::GetDefaultParameters,
				PoissonGeneratorDeviceVector::GetNeuronModelInfo,
				PoissonGeneratorDeviceVector::GetVectorizableParameters);
		NeuronModelFactory::NeuronModelMap[SinCurrentDeviceVector::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				SinCurrentDeviceVector::ParseNeuronModel,
				SinCurrentDeviceVector::CreateNeuronModel,
				SinCurrentDeviceVector::GetDefaultParameters,
				SinCurrentDeviceVector::GetNeuronModelInfo,
				SinCurrentDeviceVector::GetVectorizableParameters);
		NeuronModelFactory::NeuronModelMap[AdExTimeDrivenModel::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				AdExTimeDrivenModel::ParseNeuronModel,
				AdExTimeDrivenModel::CreateNeuronModel,
				AdExTimeDrivenModel::GetDefaultParameters,
				AdExTimeDrivenModel::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[AdExTimeDrivenModelVector::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				AdExTimeDrivenModelVector::ParseNeuronModel,
				AdExTimeDrivenModelVector::CreateNeuronModel,
				AdExTimeDrivenModelVector::GetDefaultParameters,
				AdExTimeDrivenModelVector::GetNeuronModelInfo,
				AdExTimeDrivenModelVector::GetVectorizableParameters);
		NeuronModelFactory::NeuronModelMap[EgidioGranuleCell_TimeDriven::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				EgidioGranuleCell_TimeDriven::ParseNeuronModel,
				EgidioGranuleCell_TimeDriven::CreateNeuronModel,
				EgidioGranuleCell_TimeDriven::GetDefaultParameters,
				EgidioGranuleCell_TimeDriven::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[HHTimeDrivenModel::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				HHTimeDrivenModel::ParseNeuronModel,
				HHTimeDrivenModel::CreateNeuronModel,
				HHTimeDrivenModel::GetDefaultParameters,
				HHTimeDrivenModel::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[IzhikevichTimeDrivenModel::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				IzhikevichTimeDrivenModel::ParseNeuronModel,
				IzhikevichTimeDrivenModel::CreateNeuronModel,
				IzhikevichTimeDrivenModel::GetDefaultParameters,
				IzhikevichTimeDrivenModel::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[LIFTimeDrivenModel::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				LIFTimeDrivenModel::ParseNeuronModel,
				LIFTimeDrivenModel::CreateNeuronModel,
				LIFTimeDrivenModel::GetDefaultParameters,
				LIFTimeDrivenModel::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[LIFTimeDrivenModel_IS::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				LIFTimeDrivenModel_IS::ParseNeuronModel,
				LIFTimeDrivenModel_IS::CreateNeuronModel,
				LIFTimeDrivenModel_IS::GetDefaultParameters,
				LIFTimeDrivenModel_IS::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[TimeDrivenInferiorOliveCell::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				TimeDrivenInferiorOliveCell::ParseNeuronModel,
				TimeDrivenInferiorOliveCell::CreateNeuronModel,
				TimeDrivenInferiorOliveCell::GetDefaultParameters,
				TimeDrivenInferiorOliveCell::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[TimeDrivenPurkinjeCell::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				TimeDrivenPurkinjeCell::ParseNeuronModel,
				TimeDrivenPurkinjeCell::CreateNeuronModel,
				TimeDrivenPurkinjeCell::GetDefaultParameters,
				TimeDrivenPurkinjeCell::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[TimeDrivenPurkinjeCell_IP::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				TimeDrivenPurkinjeCell_IP::ParseNeuronModel,
				TimeDrivenPurkinjeCell_IP::CreateNeuronModel,
				TimeDrivenPurkinjeCell_IP::GetDefaultParameters,
				TimeDrivenPurkinjeCell_IP::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[CompressSynchronousTableBasedModel::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				CompressSynchronousTableBasedModel::ParseNeuronModel,
				CompressSynchronousTableBasedModel::CreateNeuronModel,
				CompressSynchronousTableBasedModel::GetDefaultParameters,
				CompressSynchronousTableBasedModel::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[CompressTableBasedModel::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				CompressTableBasedModel::ParseNeuronModel,
				CompressTableBasedModel::CreateNeuronModel,
				CompressTableBasedModel::GetDefaultParameters,
				CompressTableBasedModel::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[SynchronousTableBasedModel::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				SynchronousTableBasedModel::ParseNeuronModel,
				SynchronousTableBasedModel::CreateNeuronModel,
				SynchronousTableBasedModel::GetDefaultParameters,
				SynchronousTableBasedModel::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[TableBasedModel::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				TableBasedModel::ParseNeuronModel,
				TableBasedModel::CreateNeuronModel,
				TableBasedModel::GetDefaultParameters,
				TableBasedModel::GetNeuronModelInfo);

#ifdef USE_CUDA
		NeuronModelFactory::NeuronModelMap[AdExTimeDrivenModel_GPU_C_INTERFACE::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				AdExTimeDrivenModel_GPU_C_INTERFACE::ParseNeuronModel,
				AdExTimeDrivenModel_GPU_C_INTERFACE::CreateNeuronModel,
				AdExTimeDrivenModel_GPU_C_INTERFACE::GetDefaultParameters,
				AdExTimeDrivenModel_GPU_C_INTERFACE::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::ParseNeuronModel,
				EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::CreateNeuronModel,
				EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetDefaultParameters,
				EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[HHTimeDrivenModel_GPU_C_INTERFACE::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				HHTimeDrivenModel_GPU_C_INTERFACE::ParseNeuronModel,
				HHTimeDrivenModel_GPU_C_INTERFACE::CreateNeuronModel,
				HHTimeDrivenModel_GPU_C_INTERFACE::GetDefaultParameters,
				HHTimeDrivenModel_GPU_C_INTERFACE::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[IzhikevichTimeDrivenModel_GPU_C_INTERFACE::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				IzhikevichTimeDrivenModel_GPU_C_INTERFACE::ParseNeuronModel,
				IzhikevichTimeDrivenModel_GPU_C_INTERFACE::CreateNeuronModel,
				IzhikevichTimeDrivenModel_GPU_C_INTERFACE::GetDefaultParameters,
				IzhikevichTimeDrivenModel_GPU_C_INTERFACE::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[LIFTimeDrivenModel_GPU_C_INTERFACE::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				LIFTimeDrivenModel_GPU_C_INTERFACE::ParseNeuronModel,
				LIFTimeDrivenModel_GPU_C_INTERFACE::CreateNeuronModel,
				LIFTimeDrivenModel_GPU_C_INTERFACE::GetDefaultParameters,
				LIFTimeDrivenModel_GPU_C_INTERFACE::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[LIFTimeDrivenModel_IS_GPU_C_INTERFACE::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				LIFTimeDrivenModel_IS_GPU_C_INTERFACE::ParseNeuronModel,
				LIFTimeDrivenModel_IS_GPU_C_INTERFACE::CreateNeuronModel,
				LIFTimeDrivenModel_IS_GPU_C_INTERFACE::GetDefaultParameters,
				LIFTimeDrivenModel_IS_GPU_C_INTERFACE::GetNeuronModelInfo);
		NeuronModelFactory::NeuronModelMap[TimeDrivenPurkinjeCell_GPU_C_INTERFACE::GetName()] =
			NeuronModelFactory::NeuronModelClass(
				TimeDrivenPurkinjeCell_GPU_C_INTERFACE::ParseNeuronModel,
				TimeDrivenPurkinjeCell_GPU_C_INTERFACE::CreateNeuronModel,
				TimeDrivenPurkinjeCell_GPU_C_INTERFACE::GetDefaultParameters,
				TimeDrivenPurkinjeCell_GPU_C_INTERFACE::GetNeuronModelInfo);
#endif //USE_CUDA
    }
}

NeuronModel * NeuronModelFactory::CreateNeuronModel(ModelDescription nmodelDescription) noexcept(false){
    NeuronModelFactory::InitializeNeuronModelFactory();
    // Find the rule description name in the learning rule map
    std::map<std::string, NeuronModelFactory::NeuronModelClass >::const_iterator it =
            NeuronModelFactory::NeuronModelMap.find(nmodelDescription.model_name);
    if (it==NeuronModelFactory::NeuronModelMap.end()){
        throw EDLUTException(TASK_NETWORK_LOAD_NEURON_MODELS, ERROR_NETWORK_NEURON_MODEL_TYPE,
                                 REPAIR_NETWORK_NEURON_MODEL_TYPE);
    }
    NeuronModel * nmodel = it->second.createFunc(nmodelDescription);
    return nmodel;
}

ModelDescription NeuronModelFactory::ParseNeuronModel(std::string ident, std::string ident_param) noexcept(false) {
    NeuronModelFactory::InitializeNeuronModelFactory();
    // Find the rule description name in the learning rule map
    std::map<std::string, NeuronModelFactory::NeuronModelClass >::const_iterator it =
            NeuronModelFactory::NeuronModelMap.find(ident);
    if (it == NeuronModelFactory::NeuronModelMap.end()) {
		cout << "Neuron model \"" << ident << "\" does not exist" << endl;
        throw EDLUTException(TASK_NETWORK_LOAD_NEURON_MODELS, ERROR_NETWORK_NEURON_MODEL_TYPE,
                             REPAIR_NETWORK_NEURON_MODEL_TYPE);
    }
    return it->second.parseFunc(ident_param+std::string(".cfg"));
}

std::map<std::string, boost::any> NeuronModelFactory::GetDefaultParameters(std::string ident) noexcept(false){
    NeuronModelFactory::InitializeNeuronModelFactory();
    // Find the rule description name in the learning rule map
    std::map<std::string, NeuronModelFactory::NeuronModelClass >::const_iterator it =
            NeuronModelFactory::NeuronModelMap.find(ident);
    if (it == NeuronModelFactory::NeuronModelMap.end()) {
        throw EDLUTException(TASK_NETWORK_LOAD_NEURON_MODELS, ERROR_NETWORK_NEURON_MODEL_TYPE,
                             REPAIR_NETWORK_NEURON_MODEL_TYPE);
    }
    return it->second.getDefaultParamFunc();
}

std::map<std::string, std::string> NeuronModelFactory::GetNeuronModelInfo(std::string ident) noexcept(false){
	NeuronModelFactory::InitializeNeuronModelFactory();
	// Find the rule description name in the learning rule map
	std::map<std::string, NeuronModelFactory::NeuronModelClass >::const_iterator it =
		NeuronModelFactory::NeuronModelMap.find(ident);
	if (it == NeuronModelFactory::NeuronModelMap.end()) {
		throw EDLUTException(TASK_NETWORK_LOAD_NEURON_MODELS, ERROR_NETWORK_NEURON_MODEL_TYPE,
			REPAIR_NETWORK_NEURON_MODEL_TYPE);
	}
	return it->second.getInfoFunc();
}

void NeuronModelFactory::PrintNeuronModelInfo(std::string ident) noexcept(false){
	NeuronModelFactory::InitializeNeuronModelFactory();
	// Find the rule description name in the learning rule map
	std::map<std::string, NeuronModelFactory::NeuronModelClass >::const_iterator it =
		NeuronModelFactory::NeuronModelMap.find(ident);
	if (it == NeuronModelFactory::NeuronModelMap.end()) {
		throw EDLUTException(TASK_NETWORK_LOAD_NEURON_MODELS, ERROR_NETWORK_NEURON_MODEL_TYPE,
			REPAIR_NETWORK_NEURON_MODEL_TYPE);
	}

	// Print a dictionary with the parameters
	std::map<std::string, std::string> newMap = it->second.getInfoFunc();
	cout << "- Information about neuron model " << ident << endl;
	for (std::map<std::string, std::string>::const_iterator it2 = newMap.begin(); it2 != newMap.end(); ++it2){
		if (it2->first == std::string("info")){
			std::cout << "\t- " << ident << ": " << it2->second << std::endl;
		}
	}
	for (std::map<std::string, std::string>::const_iterator it2 = newMap.begin(); it2 != newMap.end(); ++it2){
		if (it2->first != "info"){
			std::cout << "\t\t- " << it2->first << ": " << it2->second << std::endl;
		}
	}
}


std::map<std::string, std::string> NeuronModelFactory::GetVectorizableParameters(std::string ident) noexcept(false){
	NeuronModelFactory::InitializeNeuronModelFactory();
	// Find the rule description name in the learning rule map
	std::map<std::string, NeuronModelFactory::NeuronModelClass >::const_iterator it =
		NeuronModelFactory::NeuronModelMap.find(ident);
	if (it == NeuronModelFactory::NeuronModelMap.end()) {
		throw EDLUTException(TASK_NETWORK_LOAD_NEURON_MODELS, ERROR_NETWORK_NEURON_MODEL_TYPE,
			REPAIR_NETWORK_NEURON_MODEL_TYPE);
	}
	return it->second.getVectorizableParamFunc();
}


void NeuronModelFactory::PrintVectorizableParameters(std::string ident) noexcept(false){
	NeuronModelFactory::InitializeNeuronModelFactory();
	// Find the rule description name in the learning rule map
	std::map<std::string, NeuronModelFactory::NeuronModelClass >::const_iterator it =
		NeuronModelFactory::NeuronModelMap.find(ident);
	if (it == NeuronModelFactory::NeuronModelMap.end()) {
		throw EDLUTException(TASK_NETWORK_LOAD_NEURON_MODELS, ERROR_NETWORK_NEURON_MODEL_TYPE,
			REPAIR_NETWORK_NEURON_MODEL_TYPE);
	}

	// Print a dictionary with the parameters
	std::map<std::string, std::string> newMap = it->second.getVectorizableParamFunc();
	if (newMap.empty()){
		cout << "- Neuron model \"" << ident << "\" it is not vectorized." << endl;
	}
	else{
		cout << "- Neuron model \"" << ident << "\" it is vectorized:" << endl;
		for (std::map<std::string, std::string>::const_iterator it2 = newMap.begin(); it2 != newMap.end(); ++it2){
			std::cout << "\t- " << it2->first << ": " << it2->second << std::endl;
		}
	}
}