//https://gitlab.cern.ch/lhcb/Allen/blob/master/CMakeLists.txt


#include <cuda_runtime.h>
//#include <getopt.h>
//#include <iomanip>
#include <iostream>
//#include <vector>
//#include <algorithm>
using namespace std;

int main(int argc, char* argv[])
{
  int n_devices = 0;
  int rc = cudaGetDeviceCount(&n_devices);
  if (rc != cudaSuccess) {
    cudaError_t error = cudaGetLastError();
    std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    return 0;
  }


  for (int cd = 0; cd < n_devices; ++cd) {
    cudaDeviceProp dev;
    int rc = cudaGetDeviceProperties(&dev, cd);
    if (rc != cudaSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
      return 0;
    }
    else {
      cout<<" arch=compute_"<<dev.major<<dev.minor<<",code=sm_"<<dev.major<<dev.minor;
    }
  }
  return n_devices;
}
