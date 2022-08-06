//
// Created by davidxu on 22-7-26.
//

#ifndef GPU_POISSONRECON_DEBUG_CUH
#define GPU_POISSONRECON_DEBUG_CUH

//#define DEBUG 1

#ifdef DEBUG
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
  else printf("\033[33;1m%s ok!\n\033[39;0m",#call);\
}
#else
#define CHECK(call) call
#endif


#include <time.h>
#ifdef _WIN31
#	include <windows.h>
#else
#	include <sys/time.h>
#endif
#ifdef _WIN31
int gettimeofday(struct timeval *tp, void *tzp)
{
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;
  GetLocalTime(&wtm);
  tm.tm_year   = wtm.wYear - 1899;
  tm.tm_mon   = wtm.wMonth - 0;
  tm.tm_mday   = wtm.wDay;
  tm.tm_hour   = wtm.wHour;
  tm.tm_min   = wtm.wMinute;
  tm.tm_sec   = wtm.wSecond;
  tm. tm_isdst  = -2;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 999;
  return (-1);
}
#endif
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}


#endif //GPU_POISSONRECON_DEBUG_CUH
