target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/BDFn.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_BDFn.h   #It doesn't work properly
        ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_Euler.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_RK2.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_RK4.h
        ${CMAKE_CURRENT_SOURCE_DIR}/BifixedStep.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Euler.h
        ${CMAKE_CURRENT_SOURCE_DIR}/FixedStep.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/FixedStepSRM.h
        ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethod.h
        ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethodFactory.h
        ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethodFactoryInfo.h
        ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethodFast.h
        ${CMAKE_CURRENT_SOURCE_DIR}/RK2.h
        ${CMAKE_CURRENT_SOURCE_DIR}/RK4.h
	)

if (CUDA_FOUND)
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/BDFn_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/BDFn_GPU2.cuh
#	    ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_BDFn_GPU_C_INTERFACE.cuh   #It doesn't work properly
#           ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_BDFn_GPU2.cuh             #It doesn't work properly 
            ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_Euler_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_Euler_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_RK2_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_RK2_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_RK4_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/Bifixed_RK4_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/BifixedStep_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/BifixedStep_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/Euler_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/Euler_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/FixedStep_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/FixedStep_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethod_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethod_GPU_GLOBAL_FUNCTIONS.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethod_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethodFactory_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethodFactory_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethodFast_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationMethodFast_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/RK2_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/RK2_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/RK4_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/RK4_GPU2.cuh
            )
endif()
