#include <cstdio>
#include <bitset>
#include <cstdlib>
#include "Geometry.cuh"
#include "OctNode.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "PointStream.cuh"
#include "CmdLineParser.cuh"
#include "Debug.cuh"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/async/reduce.h"
#include "thrust/scan.h"
#include "thrust/sort.h"
#include "thrust/copy.h"


//#define FORCE_UNIT_NORMALS 1

// make readable to device  ?
__constant__ float EPSILON=float(1e-6);
__constant__ float ROUND_EPS=float(1e-5);
__constant__ int maxDepth=10;
__constant__ int markOffset=62;

__device__ int cnt=0;
__device__ int uniqueCount=0;

const int markOffset_h=62;

__device__ long long encodePoint(const Point3D<float>& pos,const long long& idx){
    long long key=0ll;
    Point3D<float> myCenter;
    myCenter.coords[0]=float(0.5);
    myCenter.coords[1]=float(0.5);
    myCenter.coords[2]=float(0.5);

    float myWidth=0.25f;
    for(int i=maxDepth-1;i>=0;--i){
        if(pos.coords[0] > myCenter.coords[0]) {
            key |= 1ll << (3 * i + 34);
            myCenter.coords[0] += myWidth;
        }else{
            myCenter.coords[0] -= myWidth;
        }

        if(pos.coords[1] > myCenter.coords[1]) {
            key |= 1ll << (3 * i + 33);
            myCenter.coords[1] += myWidth;
        }else{
            myCenter.coords[1] -= myWidth;
        }

        if(pos.coords[2] > myCenter.coords[2]) {
            key |= 1ll << (3 * i + 32);
            myCenter.coords[2] += myWidth;
        }else{
            myCenter.coords[2] -= myWidth;
        }
        myWidth/=2;
    }
    return key+idx;
}

__global__ void generateCode(Point3D<float> *points,long long *code,int size){
    long long stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    long long blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    long long offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(long long i=offset;i<size;i+=stride){
        code[i]= encodePoint(points[i],i);
    }
}

__global__ void generateMark(long long *code,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if(offset==0){
        code[0]|=1ll<<markOffset;
        return;
    }
    for(int i=offset;i<size;i+=stride){
        if(code[i]>>32 != code[i-1]>>32)
            code[i]|=1ll<<markOffset;
    }
}

__global__ void generateNodeNums(long long* uniqueCode,int *nodeNums,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if(offset==0){
        nodeNums[offset]=8;
        return;
    }
    for(int i=offset;i<size;i+=stride){
        if( (uniqueCode[i-1]>>32) & 7 != (uniqueCode[i]>>32) & 7){
            nodeNums[offset]=8;
        }
    }
}

__host__ void initUniqueNode(thrust::host_vector<long long>& uniqueCode,OctNode* uniqueNode,int size){
    for(int i=0;i<size;++i){
        uniqueNode[i].key= int( (uniqueCode[i] ^ (1ll<<markOffset) ) >> 32 );
    }
}


template<class T>
__host__ void initDeviceVector(thrust::device_vector<T>& vec,T *cpy,int size,int clear){
    if(clear)
        vec.clear();
    vec.resize(size);
    for(int i=0;i<size;++i)
        vec[i]=cpy[i];
}

//__global__ void generateNodeArray(OctNode *uniqueNode,int *nodeAddress, OctNode *NodeArray,int size){
//    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
//    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
//    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
//    for(int i=offset;i<size;i+=stride){
//        NodeArray[nodeAddress[i] + ]
//    }
//}

__global__ void outputCode(int *code){
    for(int i=0;i<15;++i){
        printf("%d\n",code[i]);
    }
}


struct markCompact{
    __host__ __device__
    bool operator()(const long long x){
        return x>>markOffset;
    }
};


int main() {
    char fileName[]="/home/davidxu/bunny.points.ply";

    PointStream<float>* pointStream;
    char* ext = GetFileExtension(fileName);
    if      (!strcasecmp(ext,"bnpts"))      pointStream = new BinaryPointStream<float>(fileName);
    else if (!strcasecmp(ext,"ply"))        pointStream = new PLYPointStream<float>(fileName);
    else                                    pointStream = new ASCIIPointStream<float>(fileName);

    Point3D<float> position,normal;
    Point3D<float> mx,mn;
    Point3D<float> center;
    int count=0;

    float scale=1;
    float scaleFactor=1.25;

    double st=cpuSecond();

    /**     Step 1: compute bounding box     */
    while(pointStream->nextPoint(position,normal)){
        for(int i=0;i<DIMENSION;++i){
            if(!count || position.coords[i]<mn.coords[i]) mn.coords[i]=position.coords[i];
            if(!count || position.coords[i]>mx.coords[i]) mx.coords[i]=position.coords[i];
        }
        ++count;
    }
    CHECK(cudaMemcpyToSymbol(cnt,&count,sizeof(int)));

    for(int i=0;i<DIMENSION;++i){
        if(!i || scale<mx.coords[i]-mn.coords[i]) scale=float(mx.coords[i]-mn.coords[i]);
        center.coords[i]=float(mx.coords[i]+mn.coords[i])/2;
    }
    scale*=scaleFactor;
    for(int i=0;i<DIMENSION;++i)
        center.coords[i]-=scale/2;

    thrust::host_vector<Point3D<float> > p_h(count),n_h(count);

    pointStream->reset();
    int idx=0;
    while(pointStream->nextPoint(position,normal)){
        int i;
        for(i=0;i<DIMENSION;++i)
            position.coords[i]=(position.coords[i]-center.coords[i])/scale;
        for(i=0;i<DIMENSION;++i)
            if(position.coords[i]<0 || position.coords[i]>1)
                break;
        p_h[idx]=position;

#if FORCE_UNIT_NORMALS
        float len=float(Length(normal));
        if(len>EPSILON)
            len=1.0f/len;
        len*=(2<<maxDepth);
        for(i=0;i<DIMENSION;++i)
            normal.coords[i]*=len;
#endif
        n_h[idx]=normal;
        ++idx;
    }

    double mid=cpuSecond();
    printf("Read takes:%lfs\n",mid-st);

    thrust::device_vector<Point3D<float> > p_d=p_h,n_d=n_h;

    Point3D<float> * samplePoints=thrust::raw_pointer_cast(&p_d[0]);
    Point3D<float> * sampleNormals=thrust::raw_pointer_cast(&n_d[0]);

    /**     Step 2: compute shuffled xyz key and sorting code   */
    long long *key=NULL;
    long long nByte=sizeof(long long)*count;
    CHECK(cudaMalloc((long long **)&key, nByte));
    dim3 grid=(32,32);
    dim3 block=(32,32);
    generateCode<<<grid,block>>>(samplePoints,key,count);
    cudaDeviceSynchronize();

    /**     Step 3: sort all sample points      */
    thrust::device_ptr<long long> key_ptr(key);
    thrust::sort_by_key(key_ptr,key_ptr+count,samplePoints);
//    thrust::sort(code_ptr,code_ptr+count,thrust::less<long long>());
    cudaDeviceSynchronize();

    /**     Step 4: find the unique nodes       */
    generateMark<<<grid,block>>>(key,count);
    cudaDeviceSynchronize();
    thrust::device_vector<long long> uniqueCode(count,-1);
    thrust::copy_if(key_ptr,key_ptr+count,uniqueCode.begin(),markCompact());
    cudaDeviceSynchronize();
    int uniqueCount_h=0;
    thrust::host_vector<long long> uniqueCode_h=uniqueCode;
    for(thrust::host_vector<long long>::iterator iter=uniqueCode_h.begin(); iter!=uniqueCode_h.end(); ++iter){
        if(*iter==-1)
            break;
        ++uniqueCount_h;
    }
    CHECK(cudaMemcpyToSymbol(uniqueCount,&uniqueCount_h,sizeof(int)));
    uniqueCode.resize(uniqueCount);

//    for(int i=0;i<uniqueCount_h;++i){
//        uniqueCode[i] &= ~(1<<markOffset_h);
//    }

//    nByte=sizeof(OctNode)*uniqueCount;
//    OctNode *uniqueNode_h=(OctNode *)malloc(nByte);
//    initUniqueNode(uniqueCode_h,uniqueNode_h,uniqueCount_h);

//    OctNode *uniqueNode=NULL;
//    CHECK(cudaMalloc((OctNode **)&uniqueNode,nByte));
//    CHECK(cudaMemcpy(uniqueNode,uniqueNode_h,nByte,cudaMemcpyHostToDevice));
//
//    free(uniqueNode_h);


    /**     Step 5: augment uniqueNode      */
    int *nodeNums=NULL;
    nByte=sizeof(int)*uniqueCount_h;
    CHECK(cudaMalloc((int **)&nodeNums,nByte));
    CHECK(cudaMemset(nodeNums,0,nByte));
    long long *uniqueCode_ptr=thrust::raw_pointer_cast(&uniqueCode[0]);
    generateNodeNums<<<grid,block>>>(uniqueCode_ptr,nodeNums,uniqueCount_h);
    cudaDeviceSynchronize();

    thrust::device_vector<int> nodeAddress;
    initDeviceVector(nodeAddress,nodeNums,uniqueCount_h,1);
    thrust::exclusive_scan(nodeAddress.begin(),nodeAddress.end(),nodeAddress.begin());
    cudaDeviceSynchronize();


    /**     Step 6: create NodeArrayD       */




    double ed=cpuSecond();
    printf("Numbers of points:%d\nNumbers of uniqueCode:%d\n",count,uniqueCount_h);
    printf("GPU:%lfs\n",ed-mid);

    cudaFree(key);
    cudaFree(nodeNums);
//    cudaFree(uniqueNode);
//    cudaFree(NodeArray);
}
