// Copyright 2003-2012 Intel Corporation. All Rights Reserved.
// 
// The source code contained or described herein and all documents related 
// to the source code ("Material") are owned by Intel Corporation or its
// suppliers or licensors.  Title to the Material remains with Intel Corporation
// or its suppliers and licensors.  The Material is protected by worldwide
// copyright and trade secret laws and treaty provisions.  No part of the
// Material may be used, copied, reproduced, modified, published, uploaded,
// posted, transmitted, distributed, or disclosed in any way without Intel's
// prior express written permission.
// 
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery
// of the Materials, either expressly, by implication, inducement, estoppel
// or otherwise.  Any license under such intellectual property rights must
// be express and approved by Intel in writing.


#include<omp.h>
#include<stdio.h>

/*------------------------------------------------------*/

__attribute__((target(mic))) void offload_check(void)
{
#ifdef __MIC__
  printf("Code running on coprocessor\n");
#else
  printf("Code running on host\n");
#endif
}

/*------------------------------------------------------*/

void mmul(float *a, int lda, float *b, int ldb, float *c, int ldc, int n)
{
	int iMaxThreads;

	/*send over the data to the card using the in clause, execute the code and return data to the host using inout clause*/
	#pragma offload target(mic) in(a:length(n*n)) in(b:length(n*n)) inout(c:length(n*n))
    {
		/*code to test whether or not running on the Intel(R) Xeon Phi(TM) coprocessor*/
		offload_check();

		iMaxThreads = omp_get_max_threads();

  		printf("matrix multiplication running with %d threads\n", iMaxThreads);
		#pragma omp parallel for 
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    for (int i = 0; i < n; ++i)
                        c[j * ldc + i] += a[k * lda + i] * b[j * ldb + k];
    }
}	

