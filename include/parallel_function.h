#ifndef PARALLEL_EXPONENTIAL_H_
#define PARALLEL_EXPONENTIAL_H_

// Example 12.7b. Taylor series, vectorized
#include <dvec.h> // Define vector classes (Intel)
#include <pmmintrin.h> // SSE3 required
// This function adds the elements of a vector, uses SSE3.
// (This is faster than the function add_horizontal)
static inline float add_elements(__m128 const & x) {
	__m128 s;
	s = _mm_hadd_ps(x, x);
	s = _mm_hadd_ps(s, s);
	return _mm_cvtss_f32(s);
}

float Exp32(float x) { // Approximate exp(x) for small x
	__declspec(align(16)) // align table by 16 byte
	static const float coef[32] = { // table of 1/n!
	1., 
	1./2., 
	1./6., 
	1./24., 
	1./120., 
	1./720.,
	1./5040.,
	1./40320., 
	1./362880.,
	1./3628800.,
	1./39916800.,
	1./4.790016E8,
	1./6.2270208E9, 
	1./8.71782912E10,
	1./1.307674368E12, 
	1./2.0922789888E13,
	1./355687428096000.,
	1./6402373705728000.,
	1./121645100408832000.,
	1./2432902008176640000.,
	1./51090942171709440000.,
	1./1124000727777607680000.,
	1./25852016738884976640000.,
	1./620448401733239439360000.,
	1./15511210043330985984000000.,
	1./403291461126605635584000000.,
	1./10888869450418352160768000000.,
	1./304888344611713860501504000000.,
	1./8841761993739701954543616000000.,
	1./2.6525285981219105863630848E32,
	1./8.22283865417792281772556288E33,
	1./2.6313083693369353016721801216E35};

	float x2 = x * x; // x^2
	float x4 = x2 * x2; // x^4
	// Define vectors of four floats
	F32vec4 xxn(x4, x2*x, x2, x); // x^1, x^2, x^3, x^4
	F32vec4 xx4(x4); // x^4
	F32vec4 s(0.f, 0.f, 0.f, 1.f); // initialize sum
	for (int i = 0; i < 32; i += 4) { // Loop by 4
		s += xxn * _mm_load_ps(coef+i); // s += x^n/n!
		xxn *= xx4; // next four x^n
	}
	return add_elements(s); // add the four sums
}

float Exp16(float x) { // Approximate exp(x) for small x
	__declspec(align(16)) // align table by 16 byte
	static const float coef[16] = { // table of 1/n!
	1., 
	1./2., 
	1./6., 
	1./24., 
	1./120., 
	1./720.,
	1./5040.,
	1./40320., 
	1./362880.,
	1./3628800.,
	1./39916800.,
	1./4.790016E8,
	1./6.2270208E9, 
	1./8.71782912E10,
	1./1.307674368E12, 
	1./2.0922789888E13};

	float x2 = x * x; // x^2
	float x4 = x2 * x2; // x^4
	// Define vectors of four floats
	F32vec4 xxn(x4, x2*x, x2, x); // x^1, x^2, x^3, x^4
	F32vec4 xx4(x4); // x^4
	F32vec4 s(0.f, 0.f, 0.f, 1.f); // initialize sum
	for (int i = 0; i < 16; i += 4) { // Loop by 4
		s += xxn * _mm_load_ps(coef+i); // s += x^n/n!
		xxn *= xx4; // next four x^n
	}
	return add_elements(s); // add the four sums
}

//// Example 12.7a. Taylor series
	//float Exp(float x ) { // Approximate exp(x) for small x
	//	float xn = x; // x^n
	//	float sum = 1.f; // sum, initialize to x^0/0!
	//	float nfac = 1.f; // n factorial
	//	for (int n = 1; n <= 32; n++) {
	//	sum += xn / nfac;
	//	xn *= x;
	//	nfac *= n+1;
	//	}
//	return sum;
//}

void invermat_parallel(float *a, float *ainv, int N_DifferentialNeuronState){

	float coef, element, inv_element;
	int mult4_N_DifferentialNeuronState=((N_DifferentialNeuronState+3)/4)*4;

	//__declspec(align(16)) float * local_a= new float [N_DifferentialNeuronState*mult4_N_DifferentialNeuronState]();
	//__declspec(align(16)) float * local_ainv= new float [N_DifferentialNeuronState*mult4_N_DifferentialNeuronState]();
	__declspec(align(16)) float local_a[16*16];
	__declspec(align(16)) float local_ainv[16*16];


	for (int i=0; i<N_DifferentialNeuronState; i++){
		for (int j=0; j<N_DifferentialNeuronState; j++){
			local_a[i*mult4_N_DifferentialNeuronState+j]=a[i*N_DifferentialNeuronState+j];
			local_ainv[i*mult4_N_DifferentialNeuronState+j]=0.0f;
		}
		local_ainv[i*mult4_N_DifferentialNeuronState+i]=1.0f;
	}

	//Iteraciones
	for (int s=0;s<N_DifferentialNeuronState;s++)
	{


		element=local_a[s*mult4_N_DifferentialNeuronState+s];

		//if(element==0){
		//	for(int n=s+1; n<N_DifferentialNeuronState; n++){
		//		element=local_a[n*mult4_N_DifferentialNeuronState+s];
		//		if(element!=0){
		//			for(int m=0; m<N_DifferentialNeuronState; m++){
		//				float value=local_a[n*mult4_N_DifferentialNeuronState+m];
		//				local_a[n*mult4_N_DifferentialNeuronState+m]=local_a[s*mult4_N_DifferentialNeuronState+m];
		//				local_a[s*mult4_N_DifferentialNeuronState+m]=value;

		//				value=local_ainv[n*mult4_N_DifferentialNeuronState+m];
		//				local_ainv[n*mult4_N_DifferentialNeuronState+m]=local_ainv[s*mult4_N_DifferentialNeuronState+m];
		//				local_ainv[s*mult4_N_DifferentialNeuronState+m]=value;
		//			}
		//			break;
		//		}
		//		if(n==(N_DifferentialNeuronState-1)){
		//			printf("This matrix is not invertible\n");
		//			exit(0);
		//		}
		//	
		//	}
		//}

		inv_element=1.0f/element;
		F32vec4 vector_inv_element(inv_element,inv_element,inv_element,inv_element);


		for (int j=0;j<mult4_N_DifferentialNeuronState;j+=4){
			_mm_store_ps(local_a + (s*mult4_N_DifferentialNeuronState+j) , vector_inv_element * _mm_load_ps(local_a + (s*mult4_N_DifferentialNeuronState+j)));
			_mm_store_ps(local_ainv + (s*mult4_N_DifferentialNeuronState+j) , vector_inv_element * _mm_load_ps(local_ainv + (s*mult4_N_DifferentialNeuronState+j)));
		}

		for(int i=0;i<N_DifferentialNeuronState;i++)
		{
			if (i!=s){
				coef=-local_a[i*mult4_N_DifferentialNeuronState+s];
				F32vec4 vector(coef, coef, coef, coef);
				for (int j=0;j<mult4_N_DifferentialNeuronState;j+=4){
					_mm_store_ps(local_a+(i*mult4_N_DifferentialNeuronState+j) ,_mm_add_ps(_mm_load_ps(local_a+(i*mult4_N_DifferentialNeuronState+j)), _mm_load_ps(local_a + (s*mult4_N_DifferentialNeuronState+j))*vector ));
					_mm_store_ps(local_ainv+(i*mult4_N_DifferentialNeuronState+j) ,_mm_add_ps(_mm_load_ps(local_ainv+(i*mult4_N_DifferentialNeuronState+j)), _mm_load_ps(local_ainv + (s*mult4_N_DifferentialNeuronState+j))*vector ));
				}
			}
		}

	}

	for (int i=0; i<N_DifferentialNeuronState; i++){
		for (int j=0; j<N_DifferentialNeuronState; j++){
			ainv[i*N_DifferentialNeuronState+j]=local_ainv[i*mult4_N_DifferentialNeuronState+j];
		}
	}

	//delete local_a;
	//delete local_ainv;
}


#endif /* PARALLEL_EXPONENTIAL_H_ */