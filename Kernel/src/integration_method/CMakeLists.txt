target_sources(${PROJECT_NAME}
        PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}/FixedStepSRM.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethodFactory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethodFactoryInfo.cpp
        )

if (CUDA_FOUND)
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethod_GPU_GLOBAL_FUNCTIONS.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethodFactory_GPU_C_INTERFACE.cu
            )
endif()
