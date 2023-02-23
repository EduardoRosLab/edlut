target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Current.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/EDLUTException.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/EDLUTFileException.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/EndRefractoryPeriodEvent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/InputCurrent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/InputSpike.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Interconnection.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/InternalSpike.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Neuron.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuronModelPropagationDelayStructure.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuronPropagationDelayStructure.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/PropagatedCurrent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/PropagatedSpike.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/PropagatedSpikeGroup.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Spike.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SynchronousTableBasedModelEvent.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SynchronousTableBasedModelInternalSpike.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenInternalSpike.cpp
        )

if (CUDA_FOUND)
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/Network_GPU.cu)

else()
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/Network.cpp)
endif()
