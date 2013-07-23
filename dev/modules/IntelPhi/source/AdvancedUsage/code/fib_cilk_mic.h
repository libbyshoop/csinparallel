/*
// Copyright 2013 Intel Corporation. All Rights Reserved.
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
*/

#include <sys/time.h>

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <functional>
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>
#include <math.h>

#include <cilk/cilk.h>


static inline unsigned long long
my_getticks ()
{
  struct timeval t;
  gettimeofday (&t, 0);
  return t.tv_sec * 1000000ULL + t.tv_usec;
}

static inline double
my_ticks_to_seconds (unsigned long long ticks)
{
  return ticks * 1.0e-6;
}


