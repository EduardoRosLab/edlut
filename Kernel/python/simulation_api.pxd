# STL and Boost libraries includes

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string

## Python libraries

#import numpy as np

# ModelDescription struct

cdef extern from 'simulation/NetworkDescription.h':
    struct ModelDescription:
        ModelDescription()
        string model_name
        map[string, any] param_map


# TimeScale enum

cdef extern from 'neuron_model/NeuronModel.h':
    enum TimeScale:
        SecondScale, MilisecondScale


cdef extern from "<boost/any.hpp>" namespace "boost":
    cppclass any:
        any()
        any(int)
        any(float)
        any(string)
        any(map[string,any])
        any(map[string,string])
        any(vector[int])
        any(vector[float])
        any(ModelDescription)

    T any_cast[T](any &) except +


# Simulation_API class

cdef extern from 'simulation/Simulation_API.h':
    cppclass Simulation_API:
        Simulation_API() except +
        vector[int] AddNeuronLayer(int, string, map[string, any], int, int) except +
        vector[int] AddSynapticLayer(const vector[int] &, const vector[int] &, map[string, any]) except +
        void AddExternalSpikeActivity(const vector[double] &, const vector[long] &) except +
        void AddExternalCurrentActivity(const vector[double] &, const vector[long] &, const vector[float] &) except +
        void GetSpikeActivity(vector[double] &, vector[long] &) except +
        void GetCompressedWeights(vector[int] &, vector[float] &) except +
        vector[float] GetWeights() except +
        vector[float] GetSelectedWeights(vector[int]) except +
        void Initialize() except +
        void SetSimulationParameters(map[string, any]) except +
        void RunSimulation(double) except +
        int AddLearningRule(string, map[string, any]) except +
        map[string, any] GetLearningRuleDefParams(string) except +
        map[string, any] GetNeuronModelDefParams(string) except +
        map[string, any] GetIntegrationMethodDefParams(string) except +
        map[string, any] GetLearningRuleParams(int) except +
        void SetLearningRuleParams(int, map[string, any]) except +
        #float* GetLayerState(vector[int], int&) except +
        vector[map[string, any]] GetNeuronParams(vector[int]) except +
        vector[map[string, any]] GetSpecificNeuronParams(vector[int]) except +
        void SetSpecificNeuronParams(int, map[string, any]) except +
        vector[string] GetAvailableNeuronModels() except +
        void PrintAvailableNeuronModels() except +
        map[string, string] GetNeuronModelInfo(string) except +
        void PrintNeuronModelInfo(string) except +
        vector[string] GetAvailableIntegrationMethods() except +
        void PrintAvailableIntegrationMethods() except +
        vector[string] GetAvailableLearningRules() except +
        void PrintAvailableLearningRules() except +
        map[string, string] GetVectorizableParameters(string) except +
        void PrintVectorizableParameters(string) except +
        void AddFileInputSpikeActivityDriver(string) except +
        void AddFileInputCurrentActivityDriver(string) except +
        void AddFileOutputSpikeActivityDriver(string) except +
        void AddFileOutputMonitorDriver(string, bool) except +
        void AddFileOutputWeightDriver(string, double) except +
        void SetRandomGeneratorSeed(int)
