#include <cstdio>
#include <bitset>
#include <cstdlib>
#include <unordered_map>
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
#include "Hash.cuh"
#include "PPolynomial.cuh"
#include "FunctionData.cuh"
#include "BinaryNode.cuh"
#include "ConfirmedPPolynomial.cuh"
#include "ConfirmedSparseMatrix.cuh"
#include "CG_CUDA.cuh"
#include "MarchingCubes.cuh"
#include "iomanip"

#define FORCE_UNIT_NORMALS 1


__device__ __host__ int qpow(int p,int q) {
    int res=1;
    while(q) {
        if(q&1) res *= p;
        q>>=1;
        p*=p;
    }
    return res;
}

//#define FORCE_UNIT_NORMALS 1
__global__ void outputDeviceArray(Point3D<float> *d_addr,int size) {
    printf("print array:\n");
    for(int i=0;i<size;++i) {
        printf("%f %f %f\n",d_addr[i].coords[0],d_addr[i].coords[1],d_addr[i].coords[2]);
    }
}

__global__ void outputDeviceArray(int *d_addr,int size) {
    printf("print array:\n");
    for(int i=0;i<size;++i) {
        printf("%d\n",d_addr[i]);
    }
}

__global__ void outputDeviceArray(float *d_addr,int size) {
    printf("print array:\n");
    for(int i=0;i<size;++i) {
        printf("%f\n",d_addr[i]);
    }
}

//__constant__ double EPSILON=float(1e-6);
#define EPSILON float(1e-6)
//__constant__ float ROUND_EPS=float(1e-5);
#define ROUND_EPS float(1e-5)
//__constant__ int maxDepth=9;
#define maxDepth 9
//__constant__ int markOffset=31;
#define markOffset 31
//__constant__ int resolution=1023;
#define resolution 1023

#define stackCapacity 2000

#define convTimes 2

__constant__ int LUTparent[8][27]={
        {0,1,1,3,4,4,3,4,4,9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13},
        {1,1,2,4,4,5,4,4,5,10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14},
        {3,4,4,3,4,4,6,7,7,12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16},
        {4,4,5,4,4,5,7,7,8,13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17},
        {9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13,18,19,19,21,22,22,21,22,22},
        {10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14,19,19,20,22,22,23,22,22,23},
        {12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16,21,22,22,21,22,22,24,25,25},
        {13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17,22,22,23,22,22,23,25,25,26}
};
__constant__ int LUTchild[8][27]={
        {7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7},
        {6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6},
        {5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5},
        {4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4},
        {3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3},
        {2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2},
        {1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1},
        {0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0}
};

const int markOffset_h=31;
const int maxDepth_h=9;
const int normalize=0;

int LUTparent_h[8][27]={
        {0,1,1,3,4,4,3,4,4,9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13},
        {1,1,2,4,4,5,4,4,5,10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14},
        {3,4,4,3,4,4,6,7,7,12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16},
        {4,4,5,4,4,5,7,7,8,13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17},
        {9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13,18,19,19,21,22,22,21,22,22},
        {10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14,19,19,20,22,22,23,22,22,23},
        {12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16,21,22,22,21,22,22,24,25,25},
        {13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17,22,22,23,22,22,23,25,25,26}
};
int LUTchild_h[8][27]={
        {7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7},
        {6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6},
        {5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5},
        {4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4},
        {3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3},
        {2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2},
        {1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1},
        {0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0}
};

struct markCompact{
    __device__ bool operator()(const long long &x){
        return ( x & (1ll<<markOffset) ) > 0;
    }
};

__device__ long long encodePoint(const Point3D<float> &pos,const long long& idx){
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

__global__ void generateStartHash(long long *key,KeyValue *hashTable,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        // same key[i]>>32 meet, the point at same node meet
        insertMin(hashTable,int(key[i]>>32),i);
    }
}

__global__ void generateCountHash(long long *key,KeyValue *hashTable,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        keyAdd(hashTable,int(key[i]>>32));
    }
}


__global__ void generateMark(long long *code,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if(offset==0){
        code[0] &= ~((1ll<<32)-1);
        code[0]|=1ll<<markOffset;
        offset+=stride;
    }
    for(int i=offset;i<size;i+=stride){
        code[i] &= ~((1ll<<32)-1);
        code[i] += i;
        if(code[i]>>32 != code[i-1]>>32) {
            code[i] |= 1ll << markOffset;
        }
    }
}

__global__ void initUniqueNode(long long *uniqueCode, KeyValue *keyStart,KeyValue *keyCount,
                               int *OriginIdx,OctNode *uniqueNode, int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        uniqueNode[i].key= int(uniqueCode[i] >> 32 ) ;
        uniqueNode[i].pidx=find(keyStart,uniqueNode[i].key);
        uniqueNode[i].pnum=find(keyCount,uniqueNode[i].key);
        OriginIdx[i]=int(uniqueCode[i] & ((1ll<<markOffset)-1) );
    }
}

__global__ void generateNodeNums(long long* uniqueCode,int *nodeNums,int size,int depthD){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if(offset==0){
        offset+=stride;
    }
    for(int i=offset;i<size;i+=stride){
        if( (uniqueCode[i-1]>>(32 + 3 * (maxDepth-depthD+1) ) )  != (uniqueCode[i]>>(32 + 3 * (maxDepth-depthD+1) ) ) ){
            nodeNums[i]=8;
        }
    }
}


__global__ void generateNodeArrayD(int *OriginIdx,OctNode *uniqueNode,int *nodeAddress,int *PointToNodeArrayD,OctNode *NodeArrayD,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        int idx=nodeAddress[i] + ( uniqueNode[i].key & 7);
        NodeArrayD[idx] = uniqueNode[i];
        PointToNodeArrayD[OriginIdx[i]] = idx;
    }
}

__global__ void processPointToNodeArrayD(int *PointToNodeArrayD,int count) {
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<count;i+=stride) {
        int nowIdx=i;
        int val=PointToNodeArrayD[nowIdx];
        while(val==-1) {
            --nowIdx;
            val=PointToNodeArrayD[nowIdx];
        }
        PointToNodeArrayD[i]=val;
    }
}

__global__ void initNodeArrayD_DIdxDnum(OctNode *NodeArrayD,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        NodeArrayD[i].dnum = 1;
        NodeArrayD[i].didx = i;
    }

}

__global__ void parallelSet0xff(OctNode *uniqueNode_D_1,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride) {
        uniqueNode_D_1[i].pidx=0x7fffffff;
        uniqueNode_D_1[i].didx=0x7fffffff;
    }
}

__global__ void generateNodeKeyIndexHash(OctNode *uniqueNode,int *nodeAddress,int uniqueCount,int depthD,KeyValue *keyIndexHash){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<uniqueCount;i+=stride){
        int fatherKey=uniqueNode[i].key & (~ (7<< (3 * (maxDepth-depthD) ) ) );
//        printf("key:%d\n",fatherKey);
        insertMin(keyIndexHash,fatherKey,nodeAddress[i]/8);
    }
}



__global__ void generateUniqueNodeArrayD_1(OctNode *NodeArray_D,int DSize,KeyValue *keyIndexHash,int depthD,OctNode *uniqueNodeArrayD_1,KeyValue *uniqueNode_D_1_Idx_To_NodeArray_D_Idx){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<DSize;i+=stride){
        if(NodeArray_D[i].pnum==0){
            int st = i-i%8;
            int valid_idx;
            for(int j=0;j<8;++j){
                valid_idx = st+j;
                if(NodeArray_D[valid_idx].pnum != 0){
                    break;
                }
            }
            int fatherKey=NodeArray_D[valid_idx].key & (~ (7<< (3 * (maxDepth-depthD) ) ) );
            int idx=find(keyIndexHash,fatherKey);
            if(NodeArray_D[i].dnum!=0) {
                atomicAdd(&uniqueNodeArrayD_1[idx].dnum, NodeArray_D[i].dnum);
                atomicMin(&uniqueNodeArrayD_1[idx].didx, NodeArray_D[i].didx);
            }
            continue;
        }
        int fatherKey=NodeArray_D[i].key & (~ (7<< (3 * (maxDepth-depthD) ) ) );
        int idx=find(keyIndexHash,fatherKey);
        int sonKey = ( NodeArray_D[i].key >> (3 * (maxDepth-depthD)) ) & 7;
        uniqueNodeArrayD_1[idx].key=fatherKey;
        atomicAdd(&uniqueNodeArrayD_1[idx].pnum,NodeArray_D[i].pnum);
        atomicMin(&uniqueNodeArrayD_1[idx].pidx,NodeArray_D[i].pidx);
        atomicAdd(&uniqueNodeArrayD_1[idx].dnum,NodeArray_D[i].dnum);
        atomicMin(&uniqueNodeArrayD_1[idx].didx,NodeArray_D[i].didx);
        insert(uniqueNode_D_1_Idx_To_NodeArray_D_Idx,idx,i);
        uniqueNodeArrayD_1[idx].children[sonKey]=i;
    }
}

__global__ void generateNodeNumsD_1(OctNode *uniqueNodeArrayD_1,int *NodeNums_D_1,int size,int depthD){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if(offset==0){
        offset+=stride;
    }
    for(int i=offset;i<size;i+=stride){
        if( (uniqueNodeArrayD_1[i-1].key >> (3 * (maxDepth-depthD+1) ) )  != (uniqueNodeArrayD_1[i].key >> ( 3 * (maxDepth-depthD+1) ) ) ){
            NodeNums_D_1[i]=8;
        }
    }
}

__global__ void generateNodeArrayD_1(OctNode *uniqueNodeArrayD_1,int *nodeAddressD_1,OctNode *NodeArrayD_1,int size,int depthD,KeyValue *uniqueNode_D_1_Idx_To_NodeArray_D_Idx,OctNode *NodeArray_D){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        int newIdx=nodeAddressD_1[i] + ( (uniqueNodeArrayD_1[i].key>>(3*(maxDepth-depthD+1) ) ) & 7);
        NodeArrayD_1[newIdx] = uniqueNodeArrayD_1[i];
        NodeArray_D[find(uniqueNode_D_1_Idx_To_NodeArray_D_Idx,i)].parent=newIdx;
    }
}

__host__ void pipelineUniqueNode_D_1(OctNode *uniqueNode_D,int *nodeAddress_D,int uniqueCount_D,OctNode *NodeArray_D,int allNodeNums_D,int depthD,
                                     OctNode *&uniqueNode_D_1, int &uniqueCount_D_1, KeyValue *&uniqueNode_D_1_Idx_To_NodeArray_D_Idx)
{
    if(uniqueNode_D_1_Idx_To_NodeArray_D_Idx!=NULL){
        destroy_hashtable(uniqueNode_D_1_Idx_To_NodeArray_D_Idx);
    }
    uniqueNode_D_1_Idx_To_NodeArray_D_Idx=create_hashtable();
    uniqueCount_D_1=allNodeNums_D/8;
    long long nByte = 1ll * sizeof(OctNode) * uniqueCount_D_1;
//    CHECK(cudaMalloc((OctNode **)&uniqueNode_D_1,nByte));
//    CHECK(cudaMemset(uniqueNode_D_1,0,nByte));
    KeyValue *keyIndexHash=create_hashtable();
    dim3 grid=(32,32);
    dim3 block=(32,32);
    parallelSet0xff<<<grid,block>>>(uniqueNode_D_1,uniqueCount_D_1);
    generateNodeKeyIndexHash<<<grid,block>>>(uniqueNode_D,nodeAddress_D,uniqueCount_D,depthD,keyIndexHash);
    cudaDeviceSynchronize();
    generateUniqueNodeArrayD_1<<<grid,block>>>(NodeArray_D,allNodeNums_D,keyIndexHash,depthD,uniqueNode_D_1,uniqueNode_D_1_Idx_To_NodeArray_D_Idx);
    cudaDeviceSynchronize();
    destroy_hashtable(keyIndexHash);
}

__host__ void pipelineNodeAddress_D_1(OctNode *uniqueNode_D_1,int uniqueCount_D_1,int depthD,
                                      int *&NodeAddress_D_1)
{
    dim3 grid=(32,32);
    dim3 block=(32,32);
    int *NodeNums_D_1=NULL;
    long long nByte = 1ll * sizeof(int)*uniqueCount_D_1;
    CHECK(cudaMalloc((int **)&NodeNums_D_1,nByte));
    CHECK(cudaMemset(NodeNums_D_1,0,nByte));
//    CHECK(cudaMalloc((int **)&NodeAddress_D_1,nByte));
//    CHECK(cudaMemset(NodeAddress_D_1,0,nByte));
    generateNodeNumsD_1<<<grid,block>>>(uniqueNode_D_1, NodeNums_D_1, uniqueCount_D_1, depthD-1);
    cudaDeviceSynchronize();

    thrust::device_ptr<int> NodeNums_D_1_ptr=thrust::device_pointer_cast<int>(NodeNums_D_1);
    thrust::device_ptr<int> NodeAddress_D_1_ptr=thrust::device_pointer_cast<int>(NodeAddress_D_1);
    thrust::inclusive_scan(NodeNums_D_1_ptr,NodeNums_D_1_ptr+uniqueCount_D_1,NodeAddress_D_1_ptr);
    cudaDeviceSynchronize();
    cudaFree(NodeNums_D_1);
}

__global__ void updateParentChildren(int *BaseAddressArray_d,OctNode *NodeArray,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        if(NodeArray[i].pnum == 0) continue;
        int depth;
        for(depth=0;depth<maxDepth_h;++depth){
            if(BaseAddressArray_d[depth] <= i && BaseAddressArray_d[depth+1] > i){
                break;
            }
        }
        if(i == 0){
            NodeArray[i].parent=-1;
#pragma unroll
            for(int child=0;child<8;++child){
                NodeArray[i].children[child] += BaseAddressArray_d[depth+1];
            }
        }else {
            NodeArray[i].parent += BaseAddressArray_d[depth - 1];
#pragma unroll
            for(int child=0;child<8;++child){
                if(NodeArray[i].children[child]!=0)
                    NodeArray[i].children[child] += BaseAddressArray_d[depth+1];
            }
        }
    }
}

__global__ void updateEmptyNodeInfo(int *BaseAddressArray_d,OctNode *NodeArray,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=1+8*offset;i<size;i+=8*stride){
        int nowPIdx;
        int nowDIdx;
        int validIdx;
        int commonParent;
        for(validIdx=0;validIdx<8;++validIdx){
            if(NodeArray[i+validIdx].pnum!=0){
                nowPIdx=NodeArray[i+validIdx].pidx;
                nowDIdx=NodeArray[i+validIdx].didx;
                commonParent=NodeArray[i+validIdx].parent;
                break;
            }
        }
        int depth;
        for(depth=0;depth<maxDepth_h;++depth){
            if(BaseAddressArray_d[depth] <= i && BaseAddressArray_d[depth+1] > i){
                break;
            }
        }
        int baseKey = NodeArray[i+validIdx].key - ( ( NodeArray[i+validIdx].key ) & ( 7 << (3 * (maxDepth-depth)) ) );

        for(int j=0;j<8;++j){
            int idx=i+j;
            if(NodeArray[idx].pnum==0){
                for(int k=0;k<8;++k){
                    NodeArray[idx].children[k]=-1;
                }
            }else{
                int basePos;
                for(int k=0;k<8;++k){
                    if(NodeArray[idx].children[k]>0){
                        basePos=NodeArray[idx].children[k]-k;
                        break;
                    }
                }
                for(int k=0;k<8;++k){
                    NodeArray[idx].children[k]=basePos+k;
                }
            }
            NodeArray[idx].key = baseKey + ( j << (3 * (maxDepth-depth)) );

            NodeArray[idx].pidx = nowPIdx;
            nowPIdx += NodeArray[idx].pnum;

            if(depth != maxDepth) {
                NodeArray[idx].didx = nowDIdx;
                nowDIdx += NodeArray[idx].dnum;
            }

            NodeArray[idx].parent=commonParent;

        }
    }
}

__global__ void computeNeighbor(OctNode *NodeArray,int left,int right,int depthD){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    for(int i=offset;i<right;i+=stride){
        for(int j=0;j<27;++j){
            int sonKey = ( NodeArray[i].key >> (3 * (maxDepth-depthD)) ) & 7;
            int parentIdx = NodeArray[i].parent;
            int neighParent = NodeArray[ parentIdx ].neighs[LUTparent[sonKey][j]];
            if(neighParent != -1){
                NodeArray[i].neighs[j] = NodeArray[ neighParent ].children[LUTchild[sonKey][j]];
            }else{
                NodeArray[i].neighs[j]= -1;
            }
        }
    }
}

__host__ void pipelineBuildNodeArray(char *fileName,Point3D<float> &center,float &scale,int &count,int &NodeArray_sz,
                                     int NodeArrayCount_h[maxDepth_h+1],int BaseAddressArray_h[maxDepth_h+1], //host
                                     Point3D<float> *&samplePoints_d,Point3D<float> *&sampleNormals_d,int *&PointToNodeArrayD,OctNode *&NodeArray)    //device
{
    count=0;
    PointStream<float>* pointStream;
    char* ext = GetFileExtension(fileName);
    if      (!strcasecmp(ext,"bnpts"))      pointStream = new BinaryPointStream<float>(fileName);
    else if (!strcasecmp(ext,"ply"))        pointStream = new PLYPointStream<float>(fileName);
    else                                    pointStream = new ASCIIPointStream<float>(fileName);

    Point3D<float> position,normal;
    Point3D<float> mx,mn;

    scale=1;
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

    for(int i=0;i<DIMENSION;++i){
        if(!i || scale<mx.coords[i]-mn.coords[i]) scale=float(mx.coords[i]-mn.coords[i]);
        center.coords[i]=float(mx.coords[i]+mn.coords[i])/2;
    }
    scale*=scaleFactor;
    for(int i=0;i<DIMENSION;++i)
        center.coords[i]-=scale/2;

    Point3D<float> *p_h=(Point3D<float> *)malloc(sizeof(Point3D<float>) * count);
    Point3D<float> *n_h=(Point3D<float> *)malloc(sizeof(Point3D<float>) * count);

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

    //  input process may can be optimized as GPU parallel
    double mid=cpuSecond();
    printf("Total points number:%d ,Read takes:%lfs\n",count,mid-st);

    CHECK(cudaMalloc((Point3D<float> **)&samplePoints_d,sizeof(Point3D<float>) * count));
    CHECK(cudaMemcpy(samplePoints_d,p_h,sizeof(Point3D<float>) * count, cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((Point3D<float> **)&sampleNormals_d,sizeof(Point3D<float>) * count));
    CHECK(cudaMemcpy(sampleNormals_d,n_h,sizeof(Point3D<float>) * count, cudaMemcpyHostToDevice));

    /**     Step 2: compute shuffled xyz key and sorting code   */
    long long *key=NULL;
    long long nByte = 1ll * sizeof(long long)*count;
    CHECK(cudaMalloc((long long **)&key, nByte));
    dim3 grid=(32,32);
    dim3 block=(32,32);
    generateCode<<<grid,block>>>(samplePoints_d,key,count);
    cudaDeviceSynchronize();

    long long *key_backup=NULL;
    CHECK(cudaMalloc((long long **)&key_backup, nByte));
    CHECK(cudaMemcpy(key_backup,key,nByte,cudaMemcpyDeviceToDevice));

    /**     Step 3: sort all sample points      */
    thrust::device_ptr<long long> key_ptr=thrust::device_pointer_cast<long long>(key);
    thrust::sort_by_key(key_ptr,key_ptr+count,samplePoints_d);
    cudaDeviceSynchronize();

    key_ptr=thrust::device_pointer_cast<long long>(key_backup);
    thrust::sort_by_key(key_ptr,key_ptr+count,sampleNormals_d);
    cudaDeviceSynchronize();

    cudaFree(key_backup);
    key_ptr=thrust::device_pointer_cast<long long>(key);

    KeyValue* start_hashTable=create_hashtable();
    KeyValue* count_hashTable=create_hashtable();

    generateStartHash<<<grid,block>>>(key, start_hashTable,count);
    generateCountHash<<<grid,block>>>(key, count_hashTable,count);


    /**     Step 4: find the unique nodes       */
    generateMark<<<grid,block>>>(key,count);
    cudaDeviceSynchronize();
    long long *uniqueCode=NULL;
    CHECK(cudaMalloc((long long **)&uniqueCode,sizeof(long long) * count));
    thrust::device_ptr<long long> uniqueCode_ptr=thrust::device_pointer_cast<long long>(uniqueCode);
    thrust::device_ptr<long long> uniqueCode_end=thrust::copy_if(key_ptr,key_ptr+count,uniqueCode_ptr,markCompact());
    cudaDeviceSynchronize();
    cudaFree(key);

    int uniqueCount_h=uniqueCode_end-uniqueCode_ptr;

    /**     Create uniqueNode according to uniqueCode   */
    OctNode *uniqueNode=NULL;
    nByte = 1ll * sizeof(OctNode)*uniqueCount_h;
    CHECK(cudaMalloc((OctNode **)&uniqueNode,nByte));
    CHECK(cudaMemset(uniqueNode,0,nByte));
    int *OriginIdx=NULL;
    nByte = 1ll * sizeof(int)*uniqueCount_h;
    CHECK(cudaMalloc((int**)&OriginIdx,nByte));
//    CHECK(cudaMemset(OriginIdx,-1,nByte));
    initUniqueNode<<<grid,block>>>(uniqueCode,start_hashTable,count_hashTable,
                                                   OriginIdx,uniqueNode,uniqueCount_h);
    cudaDeviceSynchronize();

    destroy_hashtable(start_hashTable);
    destroy_hashtable(count_hashTable);

    /**     Step 5: augment uniqueNode      */
    int *nodeNums=NULL;
    int *nodeAddress=NULL;
    nByte = 1ll * sizeof(int)*uniqueCount_h;
    CHECK(cudaMalloc((int **)&nodeNums,nByte));
    CHECK(cudaMemset(nodeNums,0,nByte));

    CHECK(cudaMalloc((int **)&nodeAddress,nByte));
    CHECK(cudaMemset(nodeAddress,0,nByte));

    generateNodeNums<<<grid,block>>>(uniqueCode,nodeNums,uniqueCount_h,maxDepth_h);
    cudaDeviceSynchronize();

    cudaFree(uniqueCode);

    thrust::device_ptr<int> nodeNums_ptr=thrust::device_pointer_cast<int>(nodeNums);
    thrust::device_ptr<int> nodeAddress_ptr=thrust::device_pointer_cast<int>(nodeAddress);

    thrust::inclusive_scan(nodeNums_ptr,nodeNums_ptr+uniqueCount_h,nodeAddress_ptr);
    cudaDeviceSynchronize();

    cudaFree(nodeNums);


    /**     Step 6: create NodeArrayD       */
    int lastAddr;
    CHECK(cudaMemcpy(&lastAddr,nodeAddress+uniqueCount_h-1,sizeof(int),cudaMemcpyDeviceToHost));

    int allNodeNums=lastAddr+8;
    OctNode *NodeArrayD=NULL;
    nByte = 1ll * sizeof(OctNode) * allNodeNums;
    CHECK(cudaMalloc((OctNode **)&NodeArrayD, nByte));
    CHECK(cudaMemset(NodeArrayD,0,nByte));
//    int *PointToNodeArrayD=NULL;
    nByte = 1ll * sizeof(int) * count;
    CHECK(cudaMalloc((int**)&PointToNodeArrayD,nByte));
    CHECK(cudaMemset(PointToNodeArrayD,-1,nByte));
    generateNodeArrayD<<<grid,block>>>(OriginIdx,uniqueNode,nodeAddress,PointToNodeArrayD,NodeArrayD,uniqueCount_h);
    cudaDeviceSynchronize();
    initNodeArrayD_DIdxDnum<<<grid,block>>>(NodeArrayD,allNodeNums);
    processPointToNodeArrayD<<<grid,block>>>(PointToNodeArrayD,count);
    cudaDeviceSynchronize();


//    OctNode *a=(OctNode *)malloc(sizeof(OctNode)*allNodeNums);
//    cudaMemcpy(a,NodeArrayD,sizeof(OctNode)*(allNodeNums),cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 50; ++i) {
////            if(a[i].pnum==0) continue;
//        std::cout << i << " " <<std::bitset<32>(a[i].key) << " pidx:" << a[i].pidx << " pnum:" << a[i].pnum << " parent:"
//                  << a[i].parent << " didx:"<< a[i].didx << " dnum:" << a[i].dnum << std::endl;
////            for(int k=0;k<8;++k){
////                printf("children[%d]:%d ",k,a[i].children[k]);
////            }
////            puts("");
////            for(int k=0;k<27;++k){
////                printf("neigh[%d]:%d ",k,a[i].neighs[k]);
////            }
////            puts("");
//    }

    /**     D-1     */
    memset(BaseAddressArray_h,0,sizeof(int) * (maxDepth_h+1));
    OctNode **NodeArrays=(OctNode **)malloc(sizeof(OctNode *) * (maxDepth_h+1));
    NodeArrays[maxDepth_h]=NodeArrayD;

    OctNode *uniqueNode_D=uniqueNode;
    int *NodeAddress_D=nodeAddress;
    int uniqueCount_D=uniqueCount_h;
    int allNodeNums_D=allNodeNums;
    OctNode *NodeArray_D=NodeArrayD;
    for(int depthD=maxDepth_h;depthD>=1;--depthD){
//        printf("depthD:%d %d\n",depthD,allNodeNums_D);
        NodeArrayCount_h[depthD]=allNodeNums_D;

        OctNode *uniqueNode_D_1=NULL,*NodeArray_D_1=NULL;
        int D_1Nums=allNodeNums_D/8;
        nByte = 1ll * sizeof(OctNode) * D_1Nums;
        CHECK(cudaMalloc((OctNode **)&uniqueNode_D_1,nByte));
        CHECK(cudaMemset(uniqueNode_D_1,0,nByte));
        int *NodeAddress_D_1=NULL;
        nByte = 1ll * sizeof(int) * D_1Nums;
        CHECK(cudaMalloc((int **)&NodeAddress_D_1,nByte));
        CHECK(cudaMemset(NodeAddress_D_1,0,nByte));
        int uniqueCount_D_1;
        int allNodeNums_D_1;
        KeyValue *uniqueNode_D_1_Idx_To_NodeArray_D_Idx=NULL;
        pipelineUniqueNode_D_1(uniqueNode_D,NodeAddress_D,uniqueCount_D,NodeArray_D,allNodeNums_D,depthD,
                               uniqueNode_D_1,uniqueCount_D_1,uniqueNode_D_1_Idx_To_NodeArray_D_Idx);
        pipelineNodeAddress_D_1(uniqueNode_D_1,uniqueCount_D_1,depthD,
                                NodeAddress_D_1);

        if(depthD>1) {
            int lastAddrD_1;
            CHECK(cudaMemcpy(&lastAddrD_1, NodeAddress_D_1 + uniqueCount_D_1 - 1, sizeof(int), cudaMemcpyDeviceToHost));
            allNodeNums_D_1 = lastAddrD_1 + 8;

            nByte = 1ll * sizeof(OctNode) * allNodeNums_D_1;
            CHECK(cudaMalloc((OctNode **) &NodeArray_D_1, nByte));
            CHECK(cudaMemset(NodeArray_D_1, 0, nByte));

            // update NodeArray_D's parent in this global function
            generateNodeArrayD_1<<<grid, block>>>(uniqueNode_D_1, NodeAddress_D_1, NodeArray_D_1, uniqueCount_D_1, depthD,uniqueNode_D_1_Idx_To_NodeArray_D_Idx,NodeArray_D);
            cudaDeviceSynchronize();
        }else{
            // D=1, D_1=0
            // the parent of NodeArray_D = 0, don't need to update
            allNodeNums_D_1 = 1;
            NodeArray_D_1=uniqueNode_D_1;
        }

        NodeArrays[depthD-1]=NodeArray_D_1;
        NodeArray_D=NodeArray_D_1;

//        nByte = 1ll * sizeof(OctNode) *uniqueCount_D_1;
//        OctNode *h=(OctNode*)malloc(nByte);
//        cudaMemcpy(h,uniqueNode_D_1,nByte,cudaMemcpyDeviceToHost);
//        for(int i=0;i<uniqueCount_D_1;++i){
//            std::cout<<std::bitset<32>(h[i].key)<<" pidx:"<<h[i].pidx<<" pnum:"<<h[i].pnum<<std::endl;
//        }
//        printf("depth:%d uniqueNode:%d NodeArray:%d\n",depthD,uniqueCount_D_1, allNodeNums_D_1);

        cudaFree(uniqueNode_D);
        uniqueNode_D=uniqueNode_D_1;
        cudaFree(NodeAddress_D);
        NodeAddress_D=NodeAddress_D_1;
        uniqueCount_D=uniqueCount_D_1;
        allNodeNums_D=allNodeNums_D_1;
    }

    NodeArrayCount_h[0]=1;
    for(int i=1;i<=maxDepth_h;++i){
        BaseAddressArray_h[i]=BaseAddressArray_h[i-1]+NodeArrayCount_h[i-1];
//        printf("%d %d\n",BaseAddressArray_h[i],NodeArrayCount_h[i]);
    }

    nByte = 1ll * sizeof(int)*(maxDepth_h+1);
//    int *NodeArrayCount_d=NULL;
//    CHECK(cudaMalloc((int **)&NodeArrayCount_d,nByte));
//    CHECK(cudaMemcpy(NodeArrayCount_d,NodeArrayCount_h,nByte,cudaMemcpyHostToDevice));
    int *BaseAddressArray_d=NULL;
    CHECK(cudaMalloc((int **)&BaseAddressArray_d,nByte));
    CHECK(cudaMemcpy(BaseAddressArray_d,BaseAddressArray_h,nByte,cudaMemcpyHostToDevice));

    nByte = 1ll * sizeof(OctNode)*(BaseAddressArray_h[maxDepth_h]+NodeArrayCount_h[maxDepth_h]);
//    printf("%d\n",BaseAddressArray[maxDepth_h]+NodeArrayCount_h[maxDepth_h]);
    CHECK(cudaMalloc((OctNode **)&NodeArray,nByte));
    for(int i=0;i<=maxDepth_h;++i){
        CHECK(cudaMemcpy(NodeArray+BaseAddressArray_h[i],NodeArrays[i],sizeof(OctNode) * NodeArrayCount_h[i], cudaMemcpyDeviceToDevice ));
        cudaFree(NodeArrays[i]);
    }

    NodeArray_sz=(BaseAddressArray_h[maxDepth_h]+NodeArrayCount_h[maxDepth_h]);
    updateParentChildren<<<grid,block>>>(BaseAddressArray_d,NodeArray,NodeArray_sz);
    cudaDeviceSynchronize();

    updateEmptyNodeInfo<<<grid,block>>>(BaseAddressArray_d,NodeArray,NodeArray_sz);
    cudaDeviceSynchronize();

    cudaFree(BaseAddressArray_d);

    int Node_0_Neighs[27];
    for(int i=0;i<27;++i)
        Node_0_Neighs[i]=-1;
    Node_0_Neighs[13]=0;

    CHECK(cudaMemcpy(NodeArray[0].neighs,Node_0_Neighs,sizeof(int) * 27,cudaMemcpyHostToDevice));


    for(int depth=1;depth<=maxDepth_h;++depth){
        computeNeighbor<<<grid,block>>>(NodeArray,BaseAddressArray_h[depth],BaseAddressArray_h[depth]+NodeArrayCount_h[depth],depth);
        cudaDeviceSynchronize();
    }

//    OctNode *a=(OctNode *)malloc(sizeof(OctNode)*NodeArray_sz);
//    cudaMemcpy(a,NodeArray,sizeof(OctNode)*(BaseAddressArray_h[maxDepth_h]+NodeArrayCount_h[maxDepth_h]),cudaMemcpyDeviceToHost);
//    for(int j=0;j<=2;++j) {
//        int all=0;
//        for (int i = BaseAddressArray_h[j]; i < BaseAddressArray_h[j]+10; ++i) {
////            if(a[i].pnum==0) continue;
//            all+=a[i].dnum;
//            std::cout << i << " " <<std::bitset<32>(a[i].key) << " pidx:" << a[i].pidx << " pnum:" << a[i].pnum << " parent:"
//                      << a[i].parent << " didx:"<< a[i].didx << " dnum:" << a[i].dnum << std::endl;
//            for(int k=0;k<8;++k){
//                printf("children[%d]:%d ",k,a[i].children[k]);
//            }
//            puts("");
//            for(int k=0;k<27;++k){
//                printf("neigh[%d]:%d ",k,a[i].neighs[k]);
//            }
//            puts("");
//        }
//        printf("allD:%d\n",all);
//        std::cout<<std::endl;
//    }

    double ed=cpuSecond();
    printf("GPU NodeArray build takes:%lfs\n",ed-mid);

}

__host__ __device__ int getDepth(const int& idxOfNodeArray,int *&BaseAddressArray){
    int depth=0;
#if defined(__CUDA_ARCH__)
    for(depth=0;depth<maxDepth;++depth){
        if(BaseAddressArray[depth] <= idxOfNodeArray && BaseAddressArray[depth+1] > idxOfNodeArray){
            break;
        }
    }
#elif !defined(__CUDA_ARCH__)
    for(depth=0;depth<maxDepth_h;++depth){
        if(BaseAddressArray[depth] <= idxOfNodeArray && BaseAddressArray[depth+1] > idxOfNodeArray){
            break;
        }
    }
#endif
    return depth;
}

__host__ __device__ void getFunctionIdxOfNode(const int& key,const int &depthD,int idx[3]){
    idx[0]=(1<<depthD)-1;
    idx[1]=idx[0];
    idx[2]=idx[1];
#if defined(__CUDA_ARCH__)
    for(int depth=depthD;depth >= 1;--depth){
        int sonKeyX = ( key >> (3 * (maxDepth-depth) + 2) ) & 1;
        int sonKeyY = ( key >> (3 * (maxDepth-depth) + 1) ) & 1;
        int sonKeyZ = ( key >> (3 * (maxDepth-depth)) ) & 1;
        idx[0] += sonKeyX * (1<<(depthD-depth));
        idx[1] += sonKeyY * (1<<(depthD-depth));
        idx[2] += sonKeyZ * (1<<(depthD-depth));
    }
//    if(depthD==2) {
//        printf("%d %d %d\n",idx[0],idx[1],idx[2]);
//    }
#elif !defined(__CUDA_ARCH__)
    for(int depth=depthD;depth >= 1;--depth){
        int sonKeyX = ( key >> (3 * (maxDepth_h-depth) + 2) ) & 1;
        int sonKeyY = ( key >> (3 * (maxDepth_h-depth) + 1) ) & 1;
        int sonKeyZ = ( key >> (3 * (maxDepth_h-depth)) ) & 1;
        idx[0] += sonKeyX * (1<<(depthD-depth));
        idx[1] += sonKeyY * (1<<(depthD-depth));
        idx[2] += sonKeyZ * (1<<(depthD-depth));
    }
#endif
}

__host__ __device__ void getEncodedFunctionIdxOfNode(const int& key,const int &depthD,int *idx){
#if defined(__CUDA_ARCH__)
    *idx = ((1<<depthD)-1)*(1+(1<<(maxDepth+1))+(1<<(2*(maxDepth+1))) );
    for(int depth=depthD;depth >= 1;--depth){
        int sonKeyX = ( key >> (3 * (maxDepth-depth) + 2) ) & 1;
        int sonKeyY = ( key >> (3 * (maxDepth-depth) + 1) ) & 1;
        int sonKeyZ = ( key >> (3 * (maxDepth-depth)) ) & 1;
        *idx += sonKeyX * (1<<(depthD-depth)) +
                sonKeyY * (1<<(depthD-depth)) * (1<<(maxDepth+1)) +
                sonKeyZ * (1<<(depthD-depth)) * (1<<(2*(maxDepth+1)));
    }
#elif !defined(__CUDA_ARCH__)
    *idx = ((1<<depthD)-1)*(1+(1<<maxDepth_h)+(1<<(2*maxDepth_h)) );
    for(int depth=depthD;depth >= 1;--depth){
        int sonKeyX = ( key >> (3 * (maxDepth_h-depth) + 2) ) & 1;
        int sonKeyY = ( key >> (3 * (maxDepth_h-depth) + 1) ) & 1;
        int sonKeyZ = ( key >> (3 * (maxDepth_h-depth)) ) & 1;
        *idx += sonKeyX * (1<<(depthD-depth)) +
                sonKeyY * (1<<(depthD-depth)) * (1<<maxDepth_h) +
                sonKeyZ * (1<<(depthD-depth)) * (1<<(2*maxDepth_h));
    }
#endif
}

__device__ float F_center_width_Point(const ConfirmedPPolynomial<convTimes,convTimes+2> &BaseFunctionMaxDepth_d,const Point3D<float> &center,const float &width,const Point3D<float> &point){
    ConfirmedPPolynomial<convTimes,convTimes+2> thisFunction_X = BaseFunctionMaxDepth_d.shift(center.coords[0]);
    ConfirmedPPolynomial<convTimes,convTimes+2> thisFunction_Y = BaseFunctionMaxDepth_d.shift(center.coords[1]);
    ConfirmedPPolynomial<convTimes,convTimes+2> thisFunction_Z = BaseFunctionMaxDepth_d.shift(center.coords[2]);
    float x=value(thisFunction_X,point.coords[0]);
    float y=value(thisFunction_Y,point.coords[1]);
    float z=value(thisFunction_Z,point.coords[2]);
    return x*y*z;
}

__global__ void computeVectorField(ConfirmedPPolynomial<convTimes,convTimes+2> *BaseFunctionMaxDepth_d,Point3D<float> *samplePoints_d,Point3D<float> *sampleNormals_d,OctNode *NodeArray,int left,int right,Point3D<float> *VectorField){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    for(int i=offset;i<right;i+=stride){
        int idx[3];
        float width;
        getFunctionIdxOfNode(NodeArray[i].key,maxDepth,idx);
        Point3D<float> o_c;
        BinaryNode<float>::CenterAndWidth(idx[0],o_c.coords[0],width);
        BinaryNode<float>::CenterAndWidth(idx[1],o_c.coords[1],width);
        BinaryNode<float>::CenterAndWidth(idx[2],o_c.coords[2],width);
        int IdxInMaxDepth=i-left;
        Point3D<float> val;
        for(int j=0;j<27;++j){
            int neigh=NodeArray[i].neighs[j];
            if(neigh!=-1){
                for(int k=0;k<NodeArray[neigh].pnum;++k){
                    int pointIdx=NodeArray[neigh].pidx+k;
                    float weight= F_center_width_Point(*BaseFunctionMaxDepth_d,samplePoints_d[pointIdx],width,o_c);
                    val.coords[0] += weight * sampleNormals_d[pointIdx].coords[0];
                    val.coords[1] += weight * sampleNormals_d[pointIdx].coords[1];
                    val.coords[2] += weight * sampleNormals_d[pointIdx].coords[2];
//                    VectorField[IdxInMaxDepth].coords[0] += weight * sampleNormals_d[pointIdx].coords[0];
//                    VectorField[IdxInMaxDepth].coords[1] += weight * sampleNormals_d[pointIdx].coords[1];
//                    VectorField[IdxInMaxDepth].coords[2] += weight * sampleNormals_d[pointIdx].coords[2];
                }
            }
        }
        VectorField[IdxInMaxDepth].coords[0] += val.coords[0];
        VectorField[IdxInMaxDepth].coords[1] += val.coords[1];
        VectorField[IdxInMaxDepth].coords[2] += val.coords[2];
//        printf("%d %f\n",IdxInMaxDepth,VectorField[IdxInMaxDepth].coords[0]);
    }
}

__host__ __device__ float DotProduct(const Point3D<float> &p1,const Point3D<float> &p2){
    float res=0;
    res += p1.coords[0]*p2.coords[0];
    res += p1.coords[1]*p2.coords[1];
    res += p1.coords[2]*p2.coords[2];
    return res;
}

//deprecated
__global__ void precomputeFunctionIdxOfNode(int *BaseAddressArray_d,OctNode *NodeArray,int NodeArray_sz,int *NodeIdxInFunction){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<NodeArray_sz;i+=stride){
        int depthD= getDepth(i,BaseAddressArray_d);
        getFunctionIdxOfNode(NodeArray[i].key,depthD,NodeIdxInFunction+3*i);
    }
}

__global__ void precomputeEncodedFunctionIdxOfNode(int *BaseAddressArray_d,OctNode *NodeArray,int NodeArray_sz,int *NodeIdxInFunction){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<NodeArray_sz;i+=stride){
        int depthD= getDepth(i,BaseAddressArray_d);
        getEncodedFunctionIdxOfNode(NodeArray[i].key,depthD,NodeIdxInFunction+i);
    }
}

__global__ void precomputeEncodedFunctionIdxOfNode(OctNode *NodeArray,int NodeArray_sz,
                                                   int *DepthBuffer,int *NodeIdxInFunction)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<NodeArray_sz;i+=stride){
        int depthD= DepthBuffer[i];
        getEncodedFunctionIdxOfNode(NodeArray[i].key,depthD,NodeIdxInFunction+i);
    }
}

__global__ void computeEncodedFinerNodesDivergence(int *BaseAddressArray_d, int *EncodedNodeIdxInFunction,
                                                   OctNode *NodeArray, int left, int right,
                                                   Point3D<float> *VectorField,const double *dot_F_DF,
                                                   float *Divergence) {
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    int maxD=maxDepth;
    int start_D=BaseAddressArray_d[maxD];
    int res=resolution;
    int decode_offset1=(1<<(maxD+1));
    int decode_offset2=(1<<(2*(maxD+1)));
    for(int i=offset;i<right;i+=stride) {
        double val=0;
#pragma unroll
        for(int j=0;j<27;++j){
            int neighIdx=NodeArray[i].neighs[j];
            if(neighIdx == -1) continue;
            for(int k=0;k<NodeArray[neighIdx].dnum;++k){
                int Node_D_Idx=NodeArray[neighIdx].didx + k ;
                const Point3D<float> &vo = VectorField[Node_D_Idx];

                int idxO_1[3],idxO_2[3];

                int encode_idx=EncodedNodeIdxInFunction[i];
                idxO_1[0]=encode_idx%decode_offset1;
                idxO_1[1]=(encode_idx/decode_offset1)%decode_offset1;
                idxO_1[2]=encode_idx/decode_offset2;

                encode_idx=EncodedNodeIdxInFunction[start_D + Node_D_Idx];
                idxO_2[0]=encode_idx%decode_offset1;
                idxO_2[1]=(encode_idx/decode_offset1)%decode_offset1;
                idxO_2[2]=encode_idx/decode_offset2;

                int scratch[3];
//                scratch[0] = idxO_1[0] * res + idxO_2[0];
//                scratch[1] = idxO_1[1] * res + idxO_2[1];
//                scratch[2] = idxO_1[2] * res + idxO_2[2];
                scratch[0] = idxO_1[0] + idxO_2[0] * res;
                scratch[1] = idxO_1[1] + idxO_2[1] * res;
                scratch[2] = idxO_1[2] + idxO_2[2] * res;

                Point3D<float> uo;
                uo.coords[0]=dot_F_DF[scratch[0]];
                uo.coords[1]=dot_F_DF[scratch[1]];
                uo.coords[2]=dot_F_DF[scratch[2]];
                val += DotProduct(vo,uo);
            }
        }
        Divergence[i] = val;
//        printf("%d %f\n",i,val);
    }
}

__global__ void computeCoverNums(OctNode *NodeArray,int idx,int *coverNums){
    *coverNums=0;
    for(int i=0;i<27;++i){
        int neigh=NodeArray[idx].neighs[i];
        if(neigh != -1){
            *(coverNums+i+1) = NodeArray[neigh].dnum + *(coverNums+i);
        }else{
            *(coverNums+i+1) = *(coverNums+i);
        }
    }
}

__device__ int getNeighIdx(int *&coverNums,int threadId){
    int neighIdx=0;
    for(neighIdx=0;neighIdx<27;++neighIdx){
        if(coverNums[neighIdx] <= threadId && coverNums[neighIdx+1] > threadId){
            break;
        }
    }
    return neighIdx;
}

__global__ void generateDIdxArray(OctNode *NodeArray,int idx,int *coverNums,int *DIdxArray){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int size=coverNums[27];
    for(int i=offset;i<size;i+=stride){
        int neighIdx= getNeighIdx(coverNums,i);
        int st=NodeArray[ NodeArray[idx].neighs[neighIdx] ].didx;
        DIdxArray[i]= st + i - coverNums[neighIdx];
    }
}


__global__ void computeEncodedCoarserNodesDivergence(int *DIdxArray,int coverNums,int *BaseAddressArray_d,
                                                     int *NodeIdxInFunction,
                                                     Point3D<float> *VectorField,const double *dot_F_DF,
                                                     int idx,float *divg){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int maxD=maxDepth;
    int start_D=BaseAddressArray_d[maxD];
    int res=resolution;
    int decode_offset1=(1<<(maxD+1));
    int decode_offset2=(1<<(2*(maxD+1)));

    for(int i=offset;i<coverNums;i+=stride){
        int DIdx=DIdxArray[i];
        const Point3D<float> &vo = VectorField[DIdx];

        int idxO_1[3],idxO_2[3];

        int encode_idx=NodeIdxInFunction[idx];
        idxO_1[0]=encode_idx%decode_offset1;
        idxO_1[1]=(encode_idx/decode_offset1)%decode_offset1;
        idxO_1[2]=encode_idx/decode_offset2;

        encode_idx=NodeIdxInFunction[start_D+DIdx];
        idxO_2[0]=encode_idx%decode_offset1;
        idxO_2[1]=(encode_idx/decode_offset1)%decode_offset1;
        idxO_2[2]=encode_idx/decode_offset2;

        int scratch[3];
//        scratch[0] = idxO_1[0] * res + idxO_2[0];
//        scratch[1] = idxO_1[1] * res + idxO_2[1];
//        scratch[2] = idxO_1[2] * res + idxO_2[2];
        scratch[0] = idxO_1[0] + idxO_2[0] * res;
        scratch[1] = idxO_1[1] + idxO_2[1] * res;
        scratch[2] = idxO_1[2] + idxO_2[2] * res;

        Point3D<float> uo;
        uo.coords[0]=dot_F_DF[scratch[0]];
        uo.coords[1]=dot_F_DF[scratch[1]];
        uo.coords[2]=dot_F_DF[scratch[2]];

        divg[i] = DotProduct(vo,uo);
    }
}

__device__ double GetLaplacianEntry(double *&dot_F_DF,double *&dot_F_D2F,
                                    const int (&idx)[3])
{
    double dot[3];
    dot[0]=dot_F_DF[idx[0]];
    dot[1]=dot_F_DF[idx[1]];
    dot[2]=dot_F_DF[idx[2]];
    double Entry=(
            dot_F_D2F[idx[0]]*dot[1]*dot[2]+
            dot_F_D2F[idx[1]]*dot[0]*dot[2]+
            dot_F_D2F[idx[2]]*dot[0]*dot[1]
    );
    return Entry;
}

__global__ void GenerateSingleNodeLaplacian(double *dot_F_F,double *dot_F_D2F,
                                            int *EncodedNodeIdxInFunction,OctNode *NodeArray,
                                            int left,int right,
                                            int *rowCount,int *colIdx,float *val)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int res=resolution;
    int maxD=maxDepth;
    offset+=left;
    double eps=EPSILON;

    int decode_offset1=(1<<(maxD+1));
    int decode_offset2=(1<<(2*(maxD+1)));

    for(int i=offset;i<right;i+=stride){
        int cnt=0;
        int rowIdx=i-left;
        int colStart=rowIdx * 27;

        int idxO_1[3];
        int encode_idx=EncodedNodeIdxInFunction[i];
        idxO_1[0]=encode_idx%decode_offset1;
        idxO_1[1]=(encode_idx/decode_offset1)%decode_offset1;
        idxO_1[2]=encode_idx/decode_offset2;

        for(int j=0;j<27;++j){
            int neigh=NodeArray[i].neighs[j];
            if(neigh == -1) continue;

            int colIndex=neigh-left;

            int idxO_2[3];
            encode_idx=EncodedNodeIdxInFunction[neigh];
            idxO_2[0]=encode_idx%decode_offset1;
            idxO_2[1]=(encode_idx/decode_offset1)%decode_offset1;
            idxO_2[2]=encode_idx/decode_offset2;

            int scratch[3];
            scratch[0] = idxO_1[0] * res + idxO_2[0];
            scratch[1] = idxO_1[1] * res + idxO_2[1];
            scratch[2] = idxO_1[2] * res + idxO_2[2];

            double LaplacianEntryValue= GetLaplacianEntry(dot_F_F,dot_F_D2F,scratch);
            if(LaplacianEntryValue > eps) {
                colIdx[colStart + cnt] = colIndex;
                val[colStart + cnt] = LaplacianEntryValue;
                ++cnt;
            }
        }
        rowCount[rowIdx]=cnt;
    }
}

struct validEntry{
    __device__ bool operator()(const int &x){
        return x >= 0;
    }
};



__host__ void LaplacianIteration(int *BaseAddressArray_h, int *NodeArrayCount_h, const int& nowDepth,   //host
                                 int *EncodedNodeIdxInFunction, OctNode *NodeArray, float *Divergence,//device
                                 const int &NodeArray_sz,
                                 double *dot_F_F,double *dot_F_D2F,
                                 float *&d_x)
{
    float total_time=0.0f;
    dim3 grid=(32,32);
    dim3 block(32,32);
    long long nByte;
    nByte = 1ll * sizeof(float) * NodeArray_sz;
    CHECK(cudaMallocManaged((float**)&d_x,nByte));

    // run iteration for single depth nodes
    for(int i=0;i<=maxDepth_h;++i){
        printf("Depth %d Itetation...\n",i);
        int nowDepthNodesNum=NodeArrayCount_h[i];

        int *rowCount = NULL;
        nByte = 1ll * sizeof(int) * (nowDepthNodesNum+2);
        CHECK(cudaMallocManaged((int**)&rowCount,nByte));
        CHECK(cudaMemset(rowCount,0,nByte));

        int *colIdx = NULL;
        nByte = 1ll * sizeof(int) * nowDepthNodesNum * 27;
        CHECK(cudaMallocManaged((int**)&colIdx,nByte));
        CHECK(cudaMemset(colIdx,-1,nByte));

        float *val = NULL;
        nByte = 1ll * sizeof(float) * nowDepthNodesNum * 27;
        CHECK(cudaMallocManaged((float**)&val,nByte));
//        CHECK(cudaMemset(val,0,nByte));

        GenerateSingleNodeLaplacian<<<grid,block>>>(dot_F_F,dot_F_D2F,
                                                    EncodedNodeIdxInFunction,NodeArray,
                                                    BaseAddressArray_h[i],BaseAddressArray_h[i]+nowDepthNodesNum,
                                                    rowCount + 1,colIdx,val);
        cudaDeviceSynchronize();


        thrust::device_ptr<int> rowCount_ptr=thrust::device_pointer_cast<int>(rowCount);
//        int valNums_test=thrust::reduce(rowCount_ptr + 1,rowCount_ptr+nowDepthNodesNum + 1);

        int *RowBaseAddress = NULL;
        // first address number is meaningless
        nByte = 1ll * sizeof(int) * (nowDepthNodesNum + 2);
        CHECK(cudaMallocManaged((int**)&RowBaseAddress,nByte));
        thrust::device_ptr<int> RowBaseAddress_ptr=thrust::device_pointer_cast<int>(RowBaseAddress);
//        int temp=1;
//        CHECK(cudaMemcpy(rowCount,&temp,sizeof(int),cudaMemcpyHostToDevice));
        thrust::exclusive_scan(rowCount_ptr,rowCount_ptr+nowDepthNodesNum+1,RowBaseAddress_ptr);
        cudaDeviceSynchronize();
        int valNums;
        int lastRowNum;
        CHECK(cudaMemcpy(&valNums,RowBaseAddress+nowDepthNodesNum,sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&lastRowNum,rowCount+nowDepthNodesNum,sizeof(int),cudaMemcpyDeviceToHost));
        valNums+=lastRowNum;
        CHECK(cudaMemcpy(RowBaseAddress+nowDepthNodesNum+1,&valNums,sizeof(int),cudaMemcpyHostToDevice));

//        --valNums;
//        assert(valNums == valNums_test);


        int *MergedColIdx = NULL;
        nByte = 1ll * sizeof(int) * valNums;
        CHECK(cudaMallocManaged((int**)&MergedColIdx,nByte));
        thrust::device_ptr<int> colIdx_ptr=thrust::device_pointer_cast<int>(colIdx);
        thrust::device_ptr<int> MergedColIdx_ptr=thrust::device_pointer_cast<int>(MergedColIdx);

        float *MergedVal = NULL;
        nByte = 1ll * sizeof(float) * valNums;
        CHECK(cudaMallocManaged((float**)&MergedVal,nByte));
        thrust::device_ptr<float> val_ptr=thrust::device_pointer_cast<float>(val);
        thrust::device_ptr<float> MergedVal_ptr=thrust::device_pointer_cast<float>(MergedVal);

        thrust::device_ptr<float> MergedVal_end=thrust::copy_if(val_ptr,val_ptr+nowDepthNodesNum*27,colIdx_ptr,MergedVal_ptr,validEntry());
        thrust::device_ptr<int> MergedColIdx_end=thrust::copy_if(colIdx,colIdx+nowDepthNodesNum*27,MergedColIdx_ptr,validEntry());

        assert(MergedVal_end-MergedVal_ptr == valNums);
        assert(MergedColIdx_end-MergedColIdx_ptr == valNums);

        printf("valNums:%d\n",valNums);
//        for(int j=0;j<valNums;++j){
//            printf("matrix:%f\n",MergedVal[j]);
//        }
//
//        for(int j=0;j<nowDepthNodesNum;++j){
//            printf("V:%f\n",Divergence[BaseAddressArray_h[i]+j]);
//        }
        total_time += solverCG_DeviceToDevice(nowDepthNodesNum,valNums,
                                              RowBaseAddress+1,
                                              MergedColIdx,
                                              MergedVal,
                                              Divergence+BaseAddressArray_h[i],
                                              d_x+BaseAddressArray_h[i]);

//        for(int j=0;j<nowDepthNodesNum;++j){
//            printf("X:%f\n",d_x[BaseAddressArray_h[i]+j]);
//        }

        cudaFree(rowCount);
        cudaFree(colIdx);
        cudaFree(val);
        cudaFree(RowBaseAddress);
        cudaFree(MergedColIdx);
        cudaFree(MergedVal);
    }

    printf("Pure CG solving process takes:%fms\n",total_time);
}

__global__ void calculatePointsImplicitFunctionValue(Point3D<float> *samplePoints_d,int *PointToNodeArrayD,int count,int start_D,
                                                     OctNode *NodeArray,float *d_x,
                                                     int *EncodedNodeIdxInFunction, ConfirmedPPolynomial<convTimes+1,convTimes+2> *baseFunctions_d,
                                                     float *pointValue)
{
//    printf("count:%d\n",count);
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int maxD=maxDepth;
    int decode_offset1=(1<<(maxD+1));
    int decode_offset2=(1<<(2*(maxD+1)));
    for(int i=offset;i<count;i+=stride){
        int leaveNodeIdx = start_D + PointToNodeArrayD[i];
        int nowNode = leaveNodeIdx;
        float val=0.0f;
        Point3D<float> samplePoint=samplePoints_d[i];
        while(nowNode != -1){
            for(int j=0;j<27;++j){
                int neigh = NodeArray[nowNode].neighs[j];
                if(neigh != -1){

                    int idxO[3];
                    int encode_idx=EncodedNodeIdxInFunction[neigh];
                    idxO[0]=encode_idx%decode_offset1;
                    idxO[1]=(encode_idx/decode_offset1)%decode_offset1;
                    idxO[2]=encode_idx/decode_offset2;

                    ConfirmedPPolynomial<convTimes+1,convTimes+2> funcX=baseFunctions_d[idxO[0]];
                    ConfirmedPPolynomial<convTimes+1,convTimes+2> funcY=baseFunctions_d[idxO[1]];
                    ConfirmedPPolynomial<convTimes+1,convTimes+2> funcZ=baseFunctions_d[idxO[2]];

                    val += d_x[neigh] * value(funcX,samplePoint.coords[0])
                                      * value(funcY,samplePoint.coords[1])
                                      * value(funcZ,samplePoint.coords[2]);
//                    printf("%f %f %f %f\n",d_x[neigh],value(funcX,samplePoint.coords[0]),
//                           value(funcY,samplePoint.coords[1]),
//                           value(funcZ,samplePoint.coords[2]));
//                    printf("val:%f ",val);
//                    printf("d_x:%f ",d_x[neigh]);
                }
            }
            nowNode = NodeArray[nowNode].parent;
        }
        pointValue[i]=val;
//        printf("%d: %f\n",i,val);
    }
}

// deprecated
__host__ __device__ void getNodeCenter(const int &key,Point3D<float> &myCenter){
    myCenter.coords[0]=float(0.5);
    myCenter.coords[1]=float(0.5);
    myCenter.coords[2]=float(0.5);
    float myWidth=0.25f;
    for(int i=maxDepth-1;i>=0;--i){
        if(( key >> (3 * i + 2) ) & 1)
            myCenter.coords[0] += myWidth;
        else myCenter.coords[0] -= myWidth;
        if(( key >> (3 * i + 1) ) & 1)
            myCenter.coords[1] += myWidth;
        else myCenter.coords[1] -=myWidth;
        if(( key >> (3 * i) ) & 1)
            myCenter.coords[2] += myWidth;
        else myCenter.coords[2] -=myWidth;
        myWidth/=2;
    }
}

__host__ __device__ void getNodeCenterAllDepth(const int &key,Point3D<float> &myCenter,int nowDepth){
    myCenter.coords[0]=float(0.5);
    myCenter.coords[1]=float(0.5);
    myCenter.coords[2]=float(0.5);
    float myWidth=0.25f;
    for(int i=maxDepth-1;i>=(maxDepth-nowDepth);--i){
        if(( key >> (3 * i + 2) ) & 1)
            myCenter.coords[0] += myWidth;
        else myCenter.coords[0] -= myWidth;
        if(( key >> (3 * i + 1) ) & 1)
            myCenter.coords[1] += myWidth;
        else myCenter.coords[1] -=myWidth;
        if(( key >> (3 * i) ) & 1)
            myCenter.coords[2] += myWidth;
        else myCenter.coords[2] -=myWidth;
        myWidth/=2;
    }
}


__global__ void precomputeDepthAndCenter(int *BaseAddressArray_d,OctNode *NodeArray,int NodeArray_sz,
                                         int *DepthBuffer,Point3D<float> *CenterBuffer)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<NodeArray_sz;i+=stride){
        int nowDepth = getDepth(i,BaseAddressArray_d);
        DepthBuffer[i]= nowDepth;
        Point3D<float> nowCenter;
        getNodeCenterAllDepth(NodeArray[i].key,nowCenter,nowDepth);
        CenterBuffer[i]=nowCenter;
    }
}

__global__ void initVertexOwner(OctNode *NodeArray,int left,int right,
                                VertexNode *preVertexArray,
                                int *DepthBuffer,Point3D<float> *CenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    int NodeOwnerKey[8];
    int NodeOwnerIdx[8];
    for(int i=offset;i<right;i+=stride){
        int depth = DepthBuffer[i];
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;
        Point3D<float> neighCenter[27];
        int neigh[27];
#pragma unroll
        for(int k=0;k<27;++k){
            neigh[k]=NodeArray[i].neighs[k];
            if(neigh[k] != -1){
                neighCenter[k]=CenterBuffer[neigh[k]];
            }
        }
        const Point3D<float> &nodeCenter = neighCenter[13];

        Point3D<float> vertexPos[8];
#pragma unroll
        for(int j=0;j<8;++j) {
            vertexPos[j].coords[0] = nodeCenter.coords[0] + (2 * (j & 1) - 1) * halfWidth;
            vertexPos[j].coords[1] = nodeCenter.coords[1] + (2 * ((j & 2) >> 1) - 1) * halfWidth;
            vertexPos[j].coords[2] = nodeCenter.coords[2] + (2 * ((j & 4) >> 2) - 1) * halfWidth;
        }

#pragma unroll
        for(int j=0;j<8;++j)
            NodeOwnerKey[j]=0x7fffffff;
        for(int j=0;j<8;++j){
            for(int k=0;k<27;++k){
                if(neigh[k] != -1 && SquareDistance(vertexPos[j],neighCenter[k]) < Widthsq){
                    int neighKey=NodeArray[neigh[k]].key;
                    if(NodeOwnerKey[j]>neighKey){
                        NodeOwnerKey[j]=neighKey;
                        NodeOwnerIdx[j]=neigh[k];
                    }
                }
            }
        }
#pragma unroll
        for(int j=0;j<8;++j) {
            if(NodeOwnerIdx[j] == i) {
                int vertexIdx = 8 * (i - left) + j;
                preVertexArray[vertexIdx].ownerNodeIdx = i;
                preVertexArray[vertexIdx].pos.coords[0] = vertexPos[j].coords[0] ;
                preVertexArray[vertexIdx].pos.coords[1] = vertexPos[j].coords[1] ;
                preVertexArray[vertexIdx].pos.coords[2] = vertexPos[j].coords[2] ;
                preVertexArray[vertexIdx].vertexKind = j;
                preVertexArray[vertexIdx].depth = depth;
            }
        }
    }
}

// only process vertex at maxDepth
__global__ void initSubdivideVertexOwner(int NodeArray_sz,
                                         OctNode *SubdivideArray,int left,int right,
                                         VertexNode *SubdividePreVertexArray,
                                         Point3D<float> *SubdivideArrayCenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    int NodeOwnerKey[8];
    int NodeOwnerIdx[8];
    for(int i=offset;i<right;i+=stride){
        int depth = maxDepth;
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;
        Point3D<float> neighCenter[27];
        int neigh[27];
#pragma unroll
        for(int k=0;k<27;++k){
            neigh[k]=SubdivideArray[i].neighs[k];
            if(neigh[k] != -1 && neigh[k] >= NodeArray_sz){
                neighCenter[k] = SubdivideArrayCenterBuffer[neigh[k] - NodeArray_sz];
            }
        }
        const Point3D<float> &nodeCenter = neighCenter[13];

        Point3D<float> vertexPos[8];
#pragma unroll
        for(int j=0;j<8;++j) {
            vertexPos[j].coords[0] = nodeCenter.coords[0] + (2 * (j & 1) - 1) * halfWidth;
            vertexPos[j].coords[1] = nodeCenter.coords[1] + (2 * ((j & 2) >> 1) - 1) * halfWidth;
            vertexPos[j].coords[2] = nodeCenter.coords[2] + (2 * ((j & 4) >> 2) - 1) * halfWidth;
        }

#pragma unroll
        for(int j=0;j<8;++j)
            NodeOwnerKey[j]=0x7fffffff;
        for(int j=0;j<8;++j){
            for(int k=0;k<27;++k){
                if(neigh[k] != -1 && SquareDistance(vertexPos[j],neighCenter[k]) < Widthsq){
                    int neighKey;
                    if(neigh[k] < NodeArray_sz) continue;
                    else
                        neighKey=SubdivideArray[neigh[k]-NodeArray_sz].key;
                    if(NodeOwnerKey[j]>neighKey){
                        NodeOwnerKey[j]=neighKey;
                        NodeOwnerIdx[j]=neigh[k];
                    }
                }
            }
        }
#pragma unroll
        for(int j=0;j<8;++j) {
            if(NodeOwnerIdx[j] == NodeArray_sz + i) {
                int vertexIdx = 8 * (i - left) + j;
                SubdividePreVertexArray[vertexIdx].ownerNodeIdx = NodeOwnerIdx[j];
                SubdividePreVertexArray[vertexIdx].pos.coords[0] = vertexPos[j].coords[0] ;
                SubdividePreVertexArray[vertexIdx].pos.coords[1] = vertexPos[j].coords[1] ;
                SubdividePreVertexArray[vertexIdx].pos.coords[2] = vertexPos[j].coords[2] ;
                SubdividePreVertexArray[vertexIdx].vertexKind = j;
                SubdividePreVertexArray[vertexIdx].depth = depth;
            }
        }
    }
}

// only process vertex at maxDepth
__global__ void initSubdivideVertexOwner(int NodeArray_sz,
                                         EasyOctNode *SubdivideArray,int left,int right,
                                         VertexNode *SubdividePreVertexArray,
                                         Point3D<float> *SubdivideArrayCenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    int NodeOwnerKey[8];
    int NodeOwnerIdx[8];
    for(int i=offset;i<right;i+=stride){
        int depth = maxDepth;
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;
        Point3D<float> neighCenter[27];
        int neigh[27];
#pragma unroll
        for(int k=0;k<27;++k){
            neigh[k]=SubdivideArray[i].neighs[k];
            if(neigh[k] != -1 && neigh[k] >= NodeArray_sz){
                neighCenter[k] = SubdivideArrayCenterBuffer[neigh[k] - NodeArray_sz];
            }
        }
        const Point3D<float> &nodeCenter = neighCenter[13];

        Point3D<float> vertexPos[8];
#pragma unroll
        for(int j=0;j<8;++j) {
            vertexPos[j].coords[0] = nodeCenter.coords[0] + (2 * (j & 1) - 1) * halfWidth;
            vertexPos[j].coords[1] = nodeCenter.coords[1] + (2 * ((j & 2) >> 1) - 1) * halfWidth;
            vertexPos[j].coords[2] = nodeCenter.coords[2] + (2 * ((j & 4) >> 2) - 1) * halfWidth;
        }

#pragma unroll
        for(int j=0;j<8;++j)
            NodeOwnerKey[j]=0x7fffffff;
        for(int j=0;j<8;++j){
            for(int k=0;k<27;++k){
                if(neigh[k] != -1 && SquareDistance(vertexPos[j],neighCenter[k]) < Widthsq){
                    int neighKey;
                    if(neigh[k] < NodeArray_sz) continue;
                    else
                        neighKey=SubdivideArray[neigh[k]-NodeArray_sz].key;
                    if(NodeOwnerKey[j]>neighKey){
                        NodeOwnerKey[j]=neighKey;
                        NodeOwnerIdx[j]=neigh[k];
                    }
                }
            }
        }
#pragma unroll
        for(int j=0;j<8;++j) {
            if(NodeOwnerIdx[j] == NodeArray_sz + i) {
                int vertexIdx = 8 * (i - left) + j;
                SubdividePreVertexArray[vertexIdx].ownerNodeIdx = NodeOwnerIdx[j];
                SubdividePreVertexArray[vertexIdx].pos.coords[0] = vertexPos[j].coords[0] ;
                SubdividePreVertexArray[vertexIdx].pos.coords[1] = vertexPos[j].coords[1] ;
                SubdividePreVertexArray[vertexIdx].pos.coords[2] = vertexPos[j].coords[2] ;
                SubdividePreVertexArray[vertexIdx].vertexKind = j;
                SubdividePreVertexArray[vertexIdx].depth = depth;
            }
        }
    }
}

struct validVertex{
    __device__ bool operator()(const VertexNode &x){
        return x.ownerNodeIdx > 0;
    }
};

__global__ void maintainVertexNodePointerNonAtomic(VertexNode *VertexArray,int VertexArray_sz,
                                                   OctNode *NodeArray,
                                                   int *DepthBuffer,Point3D<float> *CenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<VertexArray_sz;i+=stride){
        int owner=VertexArray[i].ownerNodeIdx;
        int depth = DepthBuffer[owner];
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;
        Point3D<float> neighCenter[27];
        Point3D<float> vertexPos=VertexArray[i].pos;

        int neigh[27];
        for(int k=0;k<27;++k){
            neigh[k]=NodeArray[owner].neighs[k];
            if(neigh[k] != -1){
                neighCenter[k]=CenterBuffer[neigh[k]];
            }
        }
        int cnt=0;
        for(int k=0;k<27;++k){
            if(neigh[k] != -1 && SquareDistance(vertexPos,neighCenter[k]) < Widthsq){
                VertexArray[i].nodes[cnt]=neigh[k];
                ++cnt;
                int idx=0;
                if(neighCenter[k].coords[0]-vertexPos.coords[0] < 0) idx|=1;
                if(neighCenter[k].coords[2]-vertexPos.coords[2] < 0) idx|=4;
                if(neighCenter[k].coords[1]-vertexPos.coords[1] < 0) {
                    if(idx & 1){
                        idx+=1;
                    }else{
                        idx+=3;
                    }
                }
                NodeArray[neigh[k]].vertices[idx] = i+1;
            }
        }
    }
}

// process only the maxDepth subdivide node
__global__ void maintainSubdivideVertexNodePointer(VertexNode *VertexArray,int VertexArray_sz,
                                                   int NodeArray_sz,
                                                   OctNode *SubdivideArray,
                                                   Point3D<float> *CenterBuffer,
                                                   Point3D<float> *SubdivideArrayCenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<VertexArray_sz;i+=stride){
        int owner=VertexArray[i].ownerNodeIdx;
        int depth = maxDepth;
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;
        Point3D<float> neighCenter[27];
        Point3D<float> vertexPos=VertexArray[i].pos;

        int neigh[27];
        for (int k = 0; k < 27; ++k) {
            neigh[k]=SubdivideArray[owner-NodeArray_sz].neighs[k];
        }
        for(int k=0;k<27;++k){
            if(neigh[k] != -1){
                if(neigh[k] < NodeArray_sz) {
                    neighCenter[k] = CenterBuffer[neigh[k]];
                } else{
                    neighCenter[k] = SubdivideArrayCenterBuffer[neigh[k] - NodeArray_sz];
                }
            }
        }
        int cnt=0;
        for(int k=0;k<27;++k){
            if(neigh[k] != -1 && SquareDistance(vertexPos,neighCenter[k]) < Widthsq){
                VertexArray[i].nodes[cnt]=neigh[k];
                ++cnt;
                int idx=0;
                if(neighCenter[k].coords[0]-vertexPos.coords[0] < 0) idx|=1;
                if(neighCenter[k].coords[2]-vertexPos.coords[2] < 0) idx|=4;
                if(neighCenter[k].coords[1]-vertexPos.coords[1] < 0) {
                    if(idx & 1){
                        idx+=1;
                    }else{
                        idx+=3;
                    }
                }
                if(neigh[k] >= NodeArray_sz) {
                    if (SubdivideArray[neigh[k] - NodeArray_sz].vertices[idx] != 0) {
                        printf("idx:%d\n%d %d\n%f %f\n%f %f\n%f %f\nrelationship error\n",idx,SubdivideArray[neigh[k] - NodeArray_sz].vertices[idx],i+1,neighCenter[k].coords[0],
                               vertexPos.coords[0],neighCenter[k].coords[1],vertexPos.coords[1],
                               neighCenter[k].coords[2],vertexPos.coords[2]);
                    }
                    SubdivideArray[neigh[k] - NodeArray_sz].vertices[idx] = i + 1;
                }
            }
        }
    }
}

// process only the maxDepth subdivide node
__global__ void maintainSubdivideVertexNodePointer(VertexNode *VertexArray,int VertexArray_sz,
                                                   int NodeArray_sz,
                                                   EasyOctNode *SubdivideArray,
                                                   Point3D<float> *CenterBuffer,
                                                   Point3D<float> *SubdivideArrayCenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<VertexArray_sz;i+=stride){
        int owner=VertexArray[i].ownerNodeIdx;
        int depth = maxDepth;
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;
        Point3D<float> neighCenter[27];
        Point3D<float> vertexPos=VertexArray[i].pos;

        int neigh[27];
        for (int k = 0; k < 27; ++k) {
            neigh[k]=SubdivideArray[owner-NodeArray_sz].neighs[k];
        }
        for(int k=0;k<27;++k){
            if(neigh[k] != -1){
                if(neigh[k] < NodeArray_sz) {
                    neighCenter[k] = CenterBuffer[neigh[k]];
                } else{
                    neighCenter[k] = SubdivideArrayCenterBuffer[neigh[k] - NodeArray_sz];
                }
            }
        }
        int cnt=0;
        for(int k=0;k<27;++k){
            if(neigh[k] != -1 && SquareDistance(vertexPos,neighCenter[k]) < Widthsq){
                VertexArray[i].nodes[cnt]=neigh[k];
                ++cnt;
                int idx=0;
                if(neighCenter[k].coords[0]-vertexPos.coords[0] < 0) idx|=1;
                if(neighCenter[k].coords[2]-vertexPos.coords[2] < 0) idx|=4;
                if(neighCenter[k].coords[1]-vertexPos.coords[1] < 0) {
                    if(idx & 1){
                        idx+=1;
                    }else{
                        idx+=3;
                    }
                }
                if(neigh[k] >= NodeArray_sz)
                    SubdivideArray[neigh[k] - NodeArray_sz].vertices[idx] = i + 1;
            }
        }
    }
}

__global__ void initEdgeArray(OctNode *NodeArray,int left,int right,
                              EdgeNode *preEdgeArray,
                              int *DepthBuffer,Point3D<float> *CenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    int NodeOwnerKey[12];
    int NodeOwnerIdx[12];
    for(int i=offset;i<right;i+=stride){
        int depth = DepthBuffer[i];
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;
        Point3D<float> neighCenter[27];
        int neigh[27];
#pragma unroll
        for(int k=0;k<27;++k){
            neigh[k]=NodeArray[i].neighs[k];
            if(neigh[k] != -1){
                neighCenter[k]=CenterBuffer[neigh[k]];
            }
        }
        const Point3D<float> &nodeCenter = neighCenter[13];
        Point3D<float> edgeCenterPos[12];
        int orientation[12];
        int off[24];
#pragma unroll
        for(int j=0;j<12;++j) {
            orientation[j] = j>>2 ;
            off[2*j] = j&1;
            off[2*j+1] = (j&2)>>1;
            int multi[3];
            int dim=2*j;
            for(int k=0;k<3;++k){
                if(orientation[j]==k){
                    multi[k]=0;
                }else{
                    multi[k]=(2 * off[dim] - 1);
                    ++dim;
                }
            }
            edgeCenterPos[j].coords[0] = nodeCenter.coords[0] + multi[0] * halfWidth;
            edgeCenterPos[j].coords[1] = nodeCenter.coords[1] + multi[1] * halfWidth;
            edgeCenterPos[j].coords[2] = nodeCenter.coords[2] + multi[2] * halfWidth;
        }

#pragma unroll
        for(int j=0;j<12;++j)
            NodeOwnerKey[j]=0x7fffffff;
        for(int j=0;j<12;++j){
            for(int k=0;k<27;++k){
                if(neigh[k] != -1 && SquareDistance(edgeCenterPos[j],neighCenter[k]) < Widthsq){
                    int neighKey=NodeArray[neigh[k]].key;
                    if(NodeOwnerKey[j]>neighKey){
                        NodeOwnerKey[j]=neighKey;
                        NodeOwnerIdx[j]=neigh[k];
                    }
                }
            }
        }
#pragma unroll
        for(int j=0;j<12;++j) {
            if(NodeOwnerIdx[j] == i) {
                int edgeIdx = 12 * (i - left) + j;
                preEdgeArray[edgeIdx].ownerNodeIdx = i;
                preEdgeArray[edgeIdx].edgeKind = j;
            }
        }
    }
}

// only use for node at maxDepth
__global__ void initSubdivideEdgeArray(OctNode *SubdivideArray,int left,int right,
                                       int NodeArray_sz,
                                       EdgeNode *SubdividePreEdgeArray,
                                       Point3D<float> *SubdivideArrayCenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    int NodeOwnerKey[12];
    int NodeOwnerIdx[12];
    for(int i=offset;i<right;i+=stride){
        int depth = maxDepth;
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;
        Point3D<float> neighCenter[27];
        int neigh[27];
#pragma unroll
        for(int k=0;k<27;++k){
            neigh[k]=SubdivideArray[i].neighs[k];
            if(neigh[k] != -1 && neigh[k] >= NodeArray_sz){
                neighCenter[k]=SubdivideArrayCenterBuffer[neigh[k] - NodeArray_sz];
            }
        }
        const Point3D<float> &nodeCenter = neighCenter[13];
        Point3D<float> edgeCenterPos[12];
        int orientation[12];
        int off[24];
#pragma unroll
        for(int j=0;j<12;++j) {
            orientation[j] = j>>2 ;
            off[2*j] = j&1;
            off[2*j+1] = (j&2)>>1;
            int multi[3];
            int dim=2*j;
            for(int k=0;k<3;++k){
                if(orientation[j]==k){
                    multi[k]=0;
                }else{
                    multi[k]=(2 * off[dim] - 1);
                    ++dim;
                }
            }
            edgeCenterPos[j].coords[0] = nodeCenter.coords[0] + multi[0] * halfWidth;
            edgeCenterPos[j].coords[1] = nodeCenter.coords[1] + multi[1] * halfWidth;
            edgeCenterPos[j].coords[2] = nodeCenter.coords[2] + multi[2] * halfWidth;
        }

#pragma unroll
        for(int j=0;j<12;++j)
            NodeOwnerKey[j]=0x7fffffff;
        for(int j=0;j<12;++j){
            for(int k=0;k<27;++k){
                if(neigh[k] != -1 && SquareDistance(edgeCenterPos[j],neighCenter[k]) < Widthsq){
                    int neighKey;
                    if(neigh[k] < NodeArray_sz) continue;
                    else
                        neighKey=SubdivideArray[neigh[k] - NodeArray_sz].key;
                    if(NodeOwnerKey[j]>neighKey){
                        NodeOwnerKey[j]=neighKey;
                        NodeOwnerIdx[j]=neigh[k];
                    }
                }
            }
        }
#pragma unroll
        for(int j=0;j<12;++j) {
            if(NodeOwnerIdx[j] == i + NodeArray_sz) {
                int edgeIdx = 12 * (i - left) + j;
                SubdividePreEdgeArray[edgeIdx].ownerNodeIdx = NodeOwnerIdx[j];
                SubdividePreEdgeArray[edgeIdx].edgeKind = j;
            }
        }
    }
}

// only use for node at maxDepth
__global__ void initSubdivideEdgeArray(EasyOctNode *SubdivideArray,int left,int right,
                                       int NodeArray_sz,
                                       EdgeNode *SubdividePreEdgeArray,
                                       Point3D<float> *SubdivideArrayCenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    int NodeOwnerKey[12];
    int NodeOwnerIdx[12];
    for(int i=offset;i<right;i+=stride){
        int depth = maxDepth;
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;
        Point3D<float> neighCenter[27];
        int neigh[27];
#pragma unroll
        for(int k=0;k<27;++k){
            neigh[k]=SubdivideArray[i].neighs[k];
            if(neigh[k] != -1 && neigh[k] >= NodeArray_sz){
                neighCenter[k]=SubdivideArrayCenterBuffer[neigh[k] - NodeArray_sz];
            }
        }
        const Point3D<float> &nodeCenter = neighCenter[13];
        Point3D<float> edgeCenterPos[12];
        int orientation[12];
        int off[24];
#pragma unroll
        for(int j=0;j<12;++j) {
            orientation[j] = j>>2 ;
            off[2*j] = j&1;
            off[2*j+1] = (j&2)>>1;
            int multi[3];
            int dim=2*j;
            for(int k=0;k<3;++k){
                if(orientation[j]==k){
                    multi[k]=0;
                }else{
                    multi[k]=(2 * off[dim] - 1);
                    ++dim;
                }
            }
            edgeCenterPos[j].coords[0] = nodeCenter.coords[0] + multi[0] * halfWidth;
            edgeCenterPos[j].coords[1] = nodeCenter.coords[1] + multi[1] * halfWidth;
            edgeCenterPos[j].coords[2] = nodeCenter.coords[2] + multi[2] * halfWidth;
        }

#pragma unroll
        for(int j=0;j<12;++j)
            NodeOwnerKey[j]=0x7fffffff;
        for(int j=0;j<12;++j){
            for(int k=0;k<27;++k){
                if(neigh[k] != -1 && SquareDistance(edgeCenterPos[j],neighCenter[k]) < Widthsq){
                    int neighKey;
                    if(neigh[k] < NodeArray_sz) continue;
                    else
                        neighKey=SubdivideArray[neigh[k] - NodeArray_sz].key;
                    if(NodeOwnerKey[j]>neighKey){
                        NodeOwnerKey[j]=neighKey;
                        NodeOwnerIdx[j]=neigh[k];
                    }
                }
            }
        }
#pragma unroll
        for(int j=0;j<12;++j) {
            if(NodeOwnerIdx[j] == i + NodeArray_sz) {
                int edgeIdx = 12 * (i - left) + j;
                SubdividePreEdgeArray[edgeIdx].ownerNodeIdx = NodeOwnerIdx[j];
                SubdividePreEdgeArray[edgeIdx].edgeKind = j;
            }
        }
    }
}

__global__ void maintainEdgeNodePointer(EdgeNode *EdgeArray,int EdgeArray_sz,
                                        OctNode *NodeArray,
                                        int *DepthBuffer,Point3D<float> *CenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<EdgeArray_sz;i+=stride){
        EdgeNode nowEdge = EdgeArray[i];
        int owner = nowEdge.ownerNodeIdx;

        int depth = DepthBuffer[owner];
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;

        Point3D<float> neighCenter[27];
        int neigh[27];
        for(int k=0;k<27;++k){
            neigh[k]=NodeArray[owner].neighs[k];
            if(neigh[k] != -1){
                neighCenter[k]=CenterBuffer[neigh[k]];
            }
        }

        const Point3D<float> &nodeCenter = neighCenter[13];
        Point3D<float> edgeCenterPos;
        int multi[3];
        int dim=0;
        int orientation = nowEdge.edgeKind>>2;
        int off[2];
        off[0] = nowEdge.edgeKind & 1;
        off[1] = (nowEdge.edgeKind & 2)>>1;
        for(int k=0;k<3;++k){
            if(orientation==k){
                multi[k]=0;
            }else{
                multi[k]=(2 * off[dim] - 1);
                ++dim;
            }
        }
        edgeCenterPos.coords[0] = nodeCenter.coords[0] + multi[0] * halfWidth;
        edgeCenterPos.coords[1] = nodeCenter.coords[1] + multi[1] * halfWidth;
        edgeCenterPos.coords[2] = nodeCenter.coords[2] + multi[2] * halfWidth;

        int cnt=0;
        for(int k=0;k<27;++k){
            if(neigh[k] != -1 && SquareDistance(edgeCenterPos,neighCenter[k]) < Widthsq){
                EdgeArray[i].nodes[cnt] = neigh[k];
                ++cnt;
                int idx=orientation<<2;
                int dim=0;
                for(int j=0;j<3;++j){
                    if(orientation!=j){
                        if(neighCenter[k].coords[j]-edgeCenterPos.coords[j] < 0) idx |= (1<<dim);
                        ++dim;
                    }
                }
                NodeArray[neigh[k]].edges[idx] = i+1;
            }
        }
    }
}

// only use for edge at maxDepth
__global__ void maintainSubdivideEdgeNodePointer(EdgeNode *EdgeArray,int EdgeArray_sz,
                                                 int NodeArray_sz,
                                                 OctNode *SubdivideArray,
                                                 Point3D<float> *CenterBuffer,
                                                 Point3D<float> *SubdivideArrayCenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<EdgeArray_sz;i+=stride){
        EdgeNode nowEdge = EdgeArray[i];
        int owner = nowEdge.ownerNodeIdx;

        int depth = maxDepth;
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;

        Point3D<float> neighCenter[27];
        int neigh[27];
        for(int k=0;k<27;++k){
            neigh[k]=SubdivideArray[owner - NodeArray_sz].neighs[k];
            if(neigh[k] != -1){
                if(neigh[k] < NodeArray_sz)
                    neighCenter[k]=CenterBuffer[neigh[k]];
                else
                    neighCenter[k]=SubdivideArrayCenterBuffer[neigh[k] - NodeArray_sz];
            }
        }

        const Point3D<float> &nodeCenter = neighCenter[13];
        Point3D<float> edgeCenterPos;
        int multi[3];
        int dim=0;
        int orientation = nowEdge.edgeKind>>2;
        int off[2];
        off[0] = nowEdge.edgeKind & 1;
        off[1] = (nowEdge.edgeKind & 2)>>1;
        for(int k=0;k<3;++k){
            if(orientation==k){
                multi[k]=0;
            }else{
                multi[k]=(2 * off[dim] - 1);
                ++dim;
            }
        }
        edgeCenterPos.coords[0] = nodeCenter.coords[0] + multi[0] * halfWidth;
        edgeCenterPos.coords[1] = nodeCenter.coords[1] + multi[1] * halfWidth;
        edgeCenterPos.coords[2] = nodeCenter.coords[2] + multi[2] * halfWidth;

        int cnt=0;
        for(int k=0;k<27;++k){
            if(neigh[k] != -1 && SquareDistance(edgeCenterPos,neighCenter[k]) < Widthsq){
                EdgeArray[i].nodes[cnt] = neigh[k];
                ++cnt;
                int idx=orientation<<2;
                int dim=0;
                for(int j=0;j<3;++j){
                    if(orientation!=j){
                        if(neighCenter[k].coords[j]-edgeCenterPos.coords[j] < 0) idx |= (1<<dim);
                        ++dim;
                    }
                }
                if(neigh[k] >= NodeArray_sz)
                    SubdivideArray[neigh[k] - NodeArray_sz].edges[idx] = i+1;
            }
        }
    }
}

// only use for edge at maxDepth
__global__ void maintainSubdivideEdgeNodePointer(EdgeNode *EdgeArray,int EdgeArray_sz,
                                                 int NodeArray_sz,
                                                 EasyOctNode *SubdivideArray,
                                                 Point3D<float> *CenterBuffer,
                                                 Point3D<float> *SubdivideArrayCenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<EdgeArray_sz;i+=stride){
        EdgeNode nowEdge = EdgeArray[i];
        int owner = nowEdge.ownerNodeIdx;

        int depth = maxDepth;
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;

        Point3D<float> neighCenter[27];
        int neigh[27];
        for(int k=0;k<27;++k){
            neigh[k]=SubdivideArray[owner - NodeArray_sz].neighs[k];
            if(neigh[k] != -1){
                if(neigh[k] < NodeArray_sz)
                    neighCenter[k]=CenterBuffer[neigh[k]];
                else
                    neighCenter[k]=SubdivideArrayCenterBuffer[neigh[k] - NodeArray_sz];
            }
        }

        const Point3D<float> &nodeCenter = neighCenter[13];
        Point3D<float> edgeCenterPos;
        int multi[3];
        int dim=0;
        int orientation = nowEdge.edgeKind>>2;
        int off[2];
        off[0] = nowEdge.edgeKind & 1;
        off[1] = (nowEdge.edgeKind & 2)>>1;
        for(int k=0;k<3;++k){
            if(orientation==k){
                multi[k]=0;
            }else{
                multi[k]=(2 * off[dim] - 1);
                ++dim;
            }
        }
        edgeCenterPos.coords[0] = nodeCenter.coords[0] + multi[0] * halfWidth;
        edgeCenterPos.coords[1] = nodeCenter.coords[1] + multi[1] * halfWidth;
        edgeCenterPos.coords[2] = nodeCenter.coords[2] + multi[2] * halfWidth;

        int cnt=0;
        for(int k=0;k<27;++k){
            if(neigh[k] != -1 && SquareDistance(edgeCenterPos,neighCenter[k]) < Widthsq){
                EdgeArray[i].nodes[cnt] = neigh[k];
                ++cnt;
                int idx=orientation<<2;
                int dim=0;
                for(int j=0;j<3;++j){
                    if(orientation!=j){
                        if(neighCenter[k].coords[j]-edgeCenterPos.coords[j] < 0) idx |= (1<<dim);
                        ++dim;
                    }
                }
                if(neigh[k] >= NodeArray_sz)
                    SubdivideArray[neigh[k] - NodeArray_sz].edges[idx] = i+1;
            }
        }
    }
}


struct validEdge{
    __device__ bool operator()(const EdgeNode &x){
        return x.ownerNodeIdx > 0;
    }
};

__device__ void pushStack(int *stack,int &top,const int &val){
    if( top == stackCapacity){
        printf("Stack full, error\n");
//        for(int i=stackCapacity-10;i<stackCapacity;++i){
//            printf("%d\n",stack[i]);
//        }
    }else{
        stack[top]=val;
        ++top;
    }
}

__device__ int popStack(int *stack,int &top){
    if(top==0){
        printf("Stack empty,error.\n");
    }
    --top;
    return stack[top];
}

__device__ int findStack(int *stack,int &top,const int &val){
    for(int i=0;i<top;++i){
        if(stack[i]==val)
            return 1;
    }
    return 0;
}

__global__ void computeVertexImplicitFunctionValue(VertexNode *VertexArray,int VertexArray_sz,
                                                   OctNode *NodeArray,float *d_x,
                                                   int *EncodedNodeIdxInFunction,ConfirmedPPolynomial<convTimes+1,convTimes+2> *baseFunctions_d,
                                                   float *vvalue,float isoValue)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int maxD=maxDepth;
    int decode_offset1=(1<<(maxD+1));
    int decode_offset2=(1<<(2*(maxD+1)));
    for(int i=offset;i<VertexArray_sz;i+=stride){
        VertexNode nowVertex = VertexArray[i];
        int depth = nowVertex.depth;
        float val=0.0f;
        int exceedChildrenId = childrenVertexKind[nowVertex.vertexKind];
        int nowNode = nowVertex.ownerNodeIdx;
        if(nowNode>0){
            while(nowNode != -1){
                for(int k=0;k<27;++k){
                    int neigh = NodeArray[nowNode].neighs[k];
                    if(neigh != -1){
                        int idxO[3];
                        int encode_idx=EncodedNodeIdxInFunction[neigh];
                        idxO[0]=encode_idx%decode_offset1;
                        idxO[1]=(encode_idx/decode_offset1)%decode_offset1;
                        idxO[2]=encode_idx/decode_offset2;

                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcX=baseFunctions_d[idxO[0]];
                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcY=baseFunctions_d[idxO[1]];
                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcZ=baseFunctions_d[idxO[2]];

                        val += d_x[neigh] * value(funcX,nowVertex.pos.coords[0])
                               * value(funcY,nowVertex.pos.coords[1])
                               * value(funcZ,nowVertex.pos.coords[2]);

                    }
                }
                nowNode = NodeArray[nowNode].parent;
            }
            nowNode = nowVertex.ownerNodeIdx;
            while(depth < maxDepth){
                ++depth;
                nowNode = NodeArray[nowNode].children[exceedChildrenId];
                if(nowNode == -1) break;
                for(int k=0;k<27;++k){
                    int neigh = NodeArray[nowNode].neighs[k];
                    if(neigh != -1){
                        int idxO[3];
                        int encode_idx=EncodedNodeIdxInFunction[neigh];
                        idxO[0]=encode_idx%decode_offset1;
                        idxO[1]=(encode_idx/decode_offset1)%decode_offset1;
                        idxO[2]=encode_idx/decode_offset2;

                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcX=baseFunctions_d[idxO[0]];
                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcY=baseFunctions_d[idxO[1]];
                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcZ=baseFunctions_d[idxO[2]];

                        val += d_x[neigh] * value(funcX,nowVertex.pos.coords[0])
                               * value(funcY,nowVertex.pos.coords[1])
                               * value(funcZ,nowVertex.pos.coords[2]);
                    }
                }
            }
        }
        vvalue[i]=val-isoValue;
    }
}

__global__ void computeSubdivideVertexImplicitFunctionValue(VertexNode *SubdivideVertexArray,int SubdivideVertexArray_sz,
                                                            EasyOctNode *SubdivideArray, int rootId,
                                                            OctNode *NodeArray,int NodeArray_sz,
                                                            float *d_x, int *EncodedNodeIdxInFunction,
                                                            ConfirmedPPolynomial<convTimes+1,convTimes+2> *baseFunctions_d,
                                                            float *SubdivideVvalue,float isoValue)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int maxD=maxDepth;
    int decode_offset1=(1<<(maxD+1));
    int decode_offset2=(1<<(2*(maxD+1)));
    for(int i=offset;i<SubdivideVertexArray_sz;i+=stride){
        VertexNode nowVertex = SubdivideVertexArray[i];
        int depth = maxDepth;
        float val=0.0f;
        int nowNode = nowVertex.ownerNodeIdx;
        if(nowNode>0){
            while(nowNode != -1){
                for(int k=0;k<27;++k){
                    int neigh;
                    if(nowNode < NodeArray_sz)
                        neigh = NodeArray[nowNode].neighs[k];
                    else
                        neigh = SubdivideArray[nowNode - NodeArray_sz].neighs[k];
                    if(neigh != -1){
                        if(neigh == NodeArray_sz)
                            neigh = rootId;
                        int idxO[3];
                        int encode_idx;
                        if(neigh < NodeArray_sz)
                            encode_idx=EncodedNodeIdxInFunction[neigh];
                        else continue;  // d_x = 0 in Subdivide space
                        idxO[0]=encode_idx%decode_offset1;
                        idxO[1]=(encode_idx/decode_offset1)%decode_offset1;
                        idxO[2]=encode_idx/decode_offset2;

                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcX=baseFunctions_d[idxO[0]];
                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcY=baseFunctions_d[idxO[1]];
                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcZ=baseFunctions_d[idxO[2]];

                        val += d_x[neigh] * value(funcX,nowVertex.pos.coords[0])
                               * value(funcY,nowVertex.pos.coords[1])
                               * value(funcZ,nowVertex.pos.coords[2]);

                    }
                }
                if(nowNode < NodeArray_sz)
                    nowNode = NodeArray[nowNode].parent;
                else
                    nowNode = SubdivideArray[nowNode - NodeArray_sz].parent;
            }
        }
        SubdivideVvalue[i]=val-isoValue;
    }
}

__global__ void computeSubdivideVertexImplicitFunctionValue(VertexNode *SubdivideVertexArray,int SubdivideVertexArray_sz,
                                                            EasyOctNode *SubdivideArray, int *ReplacedNodeId,int *IsRoot,
                                                            OctNode *NodeArray,int NodeArray_sz,
                                                            float *d_x, int *EncodedNodeIdxInFunction,
                                                            ConfirmedPPolynomial<convTimes+1,convTimes+2> *baseFunctions_d,
                                                            float *SubdivideVvalue,float isoValue)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int maxD=maxDepth;
    int decode_offset1=(1<<(maxD+1));
    int decode_offset2=(1<<(2*(maxD+1)));
    for(int i=offset;i<SubdivideVertexArray_sz;i+=stride){
        VertexNode nowVertex = SubdivideVertexArray[i];
        int depth = maxDepth;
        float val=0.0f;
        int nowNode = nowVertex.ownerNodeIdx;
        if(nowNode>0){
            while(nowNode != -1){
                for(int k=0;k<27;++k){
                    int neigh;
                    if(nowNode < NodeArray_sz)
                        neigh = NodeArray[nowNode].neighs[k];
                    else
                        neigh = SubdivideArray[nowNode - NodeArray_sz].neighs[k];
                    if(neigh != -1){
                        if(neigh >= NodeArray_sz && IsRoot[neigh - NodeArray_sz])
                            neigh = ReplacedNodeId[neigh - NodeArray_sz];
                        int idxO[3];
                        int encode_idx;
                        if(neigh < NodeArray_sz)
                            encode_idx=EncodedNodeIdxInFunction[neigh];
                        else continue;  // d_x = 0 in Subdivide space
                        idxO[0]=encode_idx%decode_offset1;
                        idxO[1]=(encode_idx/decode_offset1)%decode_offset1;
                        idxO[2]=encode_idx/decode_offset2;

                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcX=baseFunctions_d[idxO[0]];
                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcY=baseFunctions_d[idxO[1]];
                        ConfirmedPPolynomial<convTimes+1,convTimes+2> funcZ=baseFunctions_d[idxO[2]];

                        val += d_x[neigh] * value(funcX,nowVertex.pos.coords[0])
                               * value(funcY,nowVertex.pos.coords[1])
                               * value(funcZ,nowVertex.pos.coords[2]);

                    }
                }
                if(nowNode < NodeArray_sz)
                    nowNode = NodeArray[nowNode].parent;
                else
                    nowNode = SubdivideArray[nowNode - NodeArray_sz].parent;
            }
        }
        SubdivideVvalue[i]=val-isoValue;
    }
}

__device__ int VertexIndex(const int &x,const int &y,const int &z){
    int ret = x | (z<<2);
    if(y==1){
        if(ret & 1){
            ++ret;
        }else{
            ret+=3;
        }
    }
    return ret;
//    return (z<<2)|(y<<1)|x;
}

__global__ void generateVexNums(EdgeNode *EdgeArray,int EdgeArray_sz,
                                OctNode *NodeArray,float *vvalue,
                                int *vexNums)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<EdgeArray_sz;i+=stride){
        EdgeNode nowEdge=EdgeArray[i];
        int owner=nowEdge.ownerNodeIdx;
        int kind=nowEdge.edgeKind;
        int orientation=kind>>2;
        int idx[2];
        idx[0]=edgeVertex[kind][0];
        idx[1]=edgeVertex[kind][1];

        int v1=NodeArray[owner].vertices[idx[0]]-1;
        int v2=NodeArray[owner].vertices[idx[1]]-1;
        if(vvalue[v1]*vvalue[v2]<=0){
            vexNums[i]=1;
        }
    }
}

__global__ void generateSubdivideVexNums(EdgeNode *SubdivideEdgeArray,int SubdivideEdgeArray_sz,
                                         int NodeArray_sz,
                                         OctNode *SubdivideArray,float *SubdivideVvalue,
                                         int *SubdivideVexNums)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<SubdivideEdgeArray_sz;i+=stride){
        EdgeNode nowEdge=SubdivideEdgeArray[i];
        int owner=nowEdge.ownerNodeIdx - NodeArray_sz;
        int kind=nowEdge.edgeKind;
        int orientation=kind>>2;
        int idx[2];
        idx[0]=edgeVertex[kind][0];
        idx[1]=edgeVertex[kind][1];

        int v1=SubdivideArray[owner].vertices[idx[0]]-1;
        int v2=SubdivideArray[owner].vertices[idx[1]]-1;
//        printf("%d %f %d %f\n",v1,SubdivideVvalue[v1],v2,SubdivideVvalue[v2]);
        if(SubdivideVvalue[v1]*SubdivideVvalue[v2]<=0){
            SubdivideVexNums[i]=1;
        }
    }
}

__global__ void generateSubdivideVexNums(EdgeNode *SubdivideEdgeArray,int SubdivideEdgeArray_sz,
                                         int NodeArray_sz,
                                         EasyOctNode *SubdivideArray,float *SubdivideVvalue,
                                         int *SubdivideVexNums)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<SubdivideEdgeArray_sz;i+=stride){
        EdgeNode nowEdge=SubdivideEdgeArray[i];
        int owner=nowEdge.ownerNodeIdx - NodeArray_sz;
        int kind=nowEdge.edgeKind;
        int orientation=kind>>2;
        int idx[2];
        idx[0]=edgeVertex[kind][0];
        idx[1]=edgeVertex[kind][1];

        int v1=SubdivideArray[owner].vertices[idx[0]]-1;
        int v2=SubdivideArray[owner].vertices[idx[1]]-1;
//        printf("%d %f %d %f\n",v1,SubdivideVvalue[v1],v2,SubdivideVvalue[v2]);
        if(SubdivideVvalue[v1]*SubdivideVvalue[v2]<=0){
            SubdivideVexNums[i]=1;
        }
    }
}

struct validVexNums{
    __device__ bool operator()(const int &x){
        return x != 0;
    }
};


__global__ void generateTriNums(OctNode *NodeArray,
                                int left,int right,
                                float *vvalue,
                                int *triNums,int *cubeCatagory)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    for(int i=offset;i<right;i+=stride){
        OctNode nowNode=NodeArray[i];
        int nowCubeCatagory=0;
        for(int j=0;j<8;++j){
            if(vvalue[nowNode.vertices[j]-1] < 0){
                nowCubeCatagory |= 1<<j;
            }
        }
        triNums[i-left]=trianglesCount[nowCubeCatagory];
        cubeCatagory[i-left]=nowCubeCatagory;
    }
}

__global__ void generateTriNums(EasyOctNode *NodeArray,
                                int left,int right,
                                float *vvalue,
                                int *triNums,int *cubeCatagory)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    for(int i=offset;i<right;i+=stride){
        EasyOctNode nowNode=NodeArray[i];
        int nowCubeCatagory=0;
        for(int j=0;j<8;++j){
            if(vvalue[nowNode.vertices[j]-1] < 0){
                nowCubeCatagory |= 1<<j;
            }
        }
        triNums[i-left]=trianglesCount[nowCubeCatagory];
        cubeCatagory[i-left]=nowCubeCatagory;
    }
}

__device__ void interpolatePoint(const Point3D<float> &p1,const Point3D<float> &p2,
                                 const int &dim,const float &v1,const float &v2,
                                 Point3D<float> & out)
{
    for(int i=0;i<3;++i){
        if(i!=dim){
            out.coords[i]=p1.coords[i];
        }
    }
    float pivot = v1/(v1-v2);
    float another_pivot=1-pivot;
    out.coords[dim]= p2.coords[dim] * pivot + p1.coords[dim] * another_pivot;
//    out.coords[dim]=p1.coords[dim]+(p2.coords[dim]-p1.coords[dim])*pivot;
}

__global__ void generateIntersectionPoint(EdgeNode *validEdgeArray,int validEdgeArray_sz,
                                          VertexNode *VertexArray,OctNode *NodeArray,
                                          int *validVexAddress,float *vvalue,
                                          Point3D<float> *VertexBuffer)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<validEdgeArray_sz;++i){
        int owner=validEdgeArray[i].ownerNodeIdx;
        int kind=validEdgeArray[i].edgeKind;
        int orientation=kind>>2;

        int idx[2];

        idx[0]=edgeVertex[kind][0];
        idx[1]=edgeVertex[kind][1];

        int v1=NodeArray[owner].vertices[idx[0]]-1;
        int v2=NodeArray[owner].vertices[idx[1]]-1;
        Point3D<float> p1=VertexArray[v1].pos,p2=VertexArray[v2].pos;
        float f1=vvalue[v1],f2=vvalue[v2];
        Point3D<float> isoPoint;
        interpolatePoint(p1,p2,
                         orientation,f1,f2,
                         isoPoint);
        VertexBuffer[validVexAddress[i]] = isoPoint;
    }
}

__global__ void generateSubdivideIntersectionPoint(EdgeNode *SubdivideValidEdgeArray,int SubdivideValidEdgeArray_sz,
                                                   VertexNode *SubdivideVertexArray,OctNode *SubdivideArray,
                                                   int NodeArray_sz,
                                                   int *SubdivideValidVexAddress,float *SubdivideVvalue,
                                                   Point3D<float> *SubdivideVertexBuffer)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<SubdivideValidEdgeArray_sz;++i){
        int owner=SubdivideValidEdgeArray[i].ownerNodeIdx - NodeArray_sz;
        int kind=SubdivideValidEdgeArray[i].edgeKind;
        int orientation=kind>>2;

        int idx[2];

        idx[0]=edgeVertex[kind][0];
        idx[1]=edgeVertex[kind][1];

        int v1=SubdivideArray[owner].vertices[idx[0]]-1;
        int v2=SubdivideArray[owner].vertices[idx[1]]-1;
//        printf("v1:%d v2:%d ok\n",v1,v2);
        Point3D<float> p1=SubdivideVertexArray[v1].pos,p2=SubdivideVertexArray[v2].pos;
//        printf("p1 p2 ok\n");
        float f1=SubdivideVvalue[v1],f2=SubdivideVvalue[v2];
//        printf("f1 f2 ok\n");
        Point3D<float> isoPoint;
        interpolatePoint(p1,p2,
                         orientation,f1,f2,
                         isoPoint);
//        printf("interpolate ok\n");
        SubdivideVertexBuffer[SubdivideValidVexAddress[i]] = isoPoint;
//        printf("assign ok\n");
//        printf("%d\n",SubdivideValidVexAddress[i]);
    }
}


__global__ void generateSubdivideIntersectionPoint(EdgeNode *SubdivideValidEdgeArray,int SubdivideValidEdgeArray_sz,
                                                   VertexNode *SubdivideVertexArray,EasyOctNode *SubdivideArray,
                                                   int NodeArray_sz,
                                                   int *SubdivideValidVexAddress,float *SubdivideVvalue,
                                                   Point3D<float> *SubdivideVertexBuffer)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<SubdivideValidEdgeArray_sz;++i){
        int owner=SubdivideValidEdgeArray[i].ownerNodeIdx - NodeArray_sz;
        int kind=SubdivideValidEdgeArray[i].edgeKind;
        int orientation=kind>>2;

        int idx[2];

        idx[0]=edgeVertex[kind][0];
        idx[1]=edgeVertex[kind][1];

        int v1=SubdivideArray[owner].vertices[idx[0]]-1;
        int v2=SubdivideArray[owner].vertices[idx[1]]-1;
        Point3D<float> p1=SubdivideVertexArray[v1].pos,p2=SubdivideVertexArray[v2].pos;
        float f1=SubdivideVvalue[v1],f2=SubdivideVvalue[v2];
        Point3D<float> isoPoint;
        interpolatePoint(p1,p2,
                         orientation,f1,f2,
                         isoPoint);
        SubdivideVertexBuffer[SubdivideValidVexAddress[i]] = isoPoint;
//        printf("%d\n",SubdivideValidVexAddress[i]);
    }
}

__global__ void generateTrianglePos(OctNode *NodeArray,int left,int right,
                                    int *triNums,int *cubeCatagory,
                                    int *vexAddress,
                                    int *triAddress, int *TriangleBuffer,
                                    FaceNode *FaceArray,int *hasSurfaceIntersection)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    for(int i=offset;i<right;i+=stride){
        OctNode nowNode = NodeArray[i];
        int depthDIdx = i-left;
        int nowTriNum = triNums[depthDIdx];
        int nowCubeCatagory = cubeCatagory[depthDIdx];
        int nowTriangleBufferStart = 3 * triAddress[depthDIdx];
        int edgeHasVertex[12]={0};
        for(int j=0;j<12;++j){
            edgeHasVertex[j]=0;
        }
        for(int j=0;j<3*nowTriNum;j+=3){
            int edgeIdx[3];
            edgeIdx[0]=triangles[nowCubeCatagory][j];
            edgeIdx[1]=triangles[nowCubeCatagory][j+1];
            edgeIdx[2]=triangles[nowCubeCatagory][j+2];

            edgeHasVertex[edgeIdx[0]]=1;
            edgeHasVertex[edgeIdx[1]]=1;
            edgeHasVertex[edgeIdx[2]]=1;

            int vertexIdx[3];
            vertexIdx[0] = vexAddress[nowNode.edges[edgeIdx[0]] - 1];
            vertexIdx[1] = vexAddress[nowNode.edges[edgeIdx[1]] - 1];
            vertexIdx[2] = vexAddress[nowNode.edges[edgeIdx[2]] - 1];

            TriangleBuffer[ nowTriangleBufferStart + j ] = vertexIdx[0];
            TriangleBuffer[ nowTriangleBufferStart + j + 1 ] = vertexIdx[1];
            TriangleBuffer[ nowTriangleBufferStart + j + 2 ] = vertexIdx[2];
        }
        int nowFace;
        int parentNodeId;
        for(int j=0;j<6;++j){
            int mark=0;
            for(int k=0;k<4;++k){
                mark |= edgeHasVertex[faceEdges[j][k]];
            }
            if(mark){
                parentNodeId=NodeArray[i].parent;
                nowFace=nowNode.faces[j] - 1;
                hasSurfaceIntersection[nowFace]=1;
                while(FaceArray[nowFace].hasParentFace != -1){
                    nowFace=NodeArray[parentNodeId].faces[j] - 1;
                    parentNodeId = NodeArray[parentNodeId].parent;
                    hasSurfaceIntersection[nowFace]=1;
                }
            }
        }
    }
}

__global__ void generateSubdivideTrianglePos(OctNode *SubdivideArray,int left,int right,
                                             int *SubdivideTriNums,int *SubdivideCubeCatagory,
                                             int *SubdivideVexAddress,
                                             int *SubdivideTriAddress, int *SubdivideTriangleBuffer)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    for(int i=offset;i<right;i+=stride){
        OctNode nowNode = SubdivideArray[i];
        int depthDIdx = i-left;
        int nowTriNum = SubdivideTriNums[depthDIdx];
        int nowCubeCatagory = SubdivideCubeCatagory[depthDIdx];
        int nowTriangleBufferStart = 3 * SubdivideTriAddress[depthDIdx];
//        printf("%d %d\n",depthDIdx,nowTriangleBufferStart);
        for(int j=0;j<3*nowTriNum;j+=3){
            int edgeIdx[3];
            edgeIdx[0]=triangles[nowCubeCatagory][j];
            edgeIdx[1]=triangles[nowCubeCatagory][j+1];
            edgeIdx[2]=triangles[nowCubeCatagory][j+2];

            int vertexIdx[3];
            vertexIdx[0] = SubdivideVexAddress[nowNode.edges[edgeIdx[0]] - 1];
            vertexIdx[1] = SubdivideVexAddress[nowNode.edges[edgeIdx[1]] - 1];
            vertexIdx[2] = SubdivideVexAddress[nowNode.edges[edgeIdx[2]] - 1];

            SubdivideTriangleBuffer[ nowTriangleBufferStart + j ] = vertexIdx[0];
            SubdivideTriangleBuffer[ nowTriangleBufferStart + j + 1 ] = vertexIdx[1];
            SubdivideTriangleBuffer[ nowTriangleBufferStart + j + 2 ] = vertexIdx[2];
        }
    }
}

__global__ void generateSubdivideTrianglePos(EasyOctNode *SubdivideArray,int left,int right,
                                             int *SubdivideTriNums,int *SubdivideCubeCatagory,
                                             int *SubdivideVexAddress,
                                             int *SubdivideTriAddress, int *SubdivideTriangleBuffer)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    for(int i=offset;i<right;i+=stride){
        EasyOctNode nowNode = SubdivideArray[i];
        int depthDIdx = i-left;
        int nowTriNum = SubdivideTriNums[depthDIdx];
        int nowCubeCatagory = SubdivideCubeCatagory[depthDIdx];
        int nowTriangleBufferStart = 3 * SubdivideTriAddress[depthDIdx];
//        printf("%d %d\n",depthDIdx,nowTriangleBufferStart);
        for(int j=0;j<3*nowTriNum;j+=3){
            int edgeIdx[3];
            edgeIdx[0]=triangles[nowCubeCatagory][j];
            edgeIdx[1]=triangles[nowCubeCatagory][j+1];
            edgeIdx[2]=triangles[nowCubeCatagory][j+2];

            int vertexIdx[3];
            vertexIdx[0] = SubdivideVexAddress[nowNode.edges[edgeIdx[0]] - 1];
            vertexIdx[1] = SubdivideVexAddress[nowNode.edges[edgeIdx[1]] - 1];
            vertexIdx[2] = SubdivideVexAddress[nowNode.edges[edgeIdx[2]] - 1];

            SubdivideTriangleBuffer[ nowTriangleBufferStart + j ] = vertexIdx[0];
            SubdivideTriangleBuffer[ nowTriangleBufferStart + j + 1 ] = vertexIdx[1];
            SubdivideTriangleBuffer[ nowTriangleBufferStart + j + 2 ] = vertexIdx[2];
        }
    }
}

__global__ void initFaceArray(OctNode *NodeArray,int left,int right,FaceNode *preFaceArray,int *DepthBuffer,Point3D<float> *CenterBuffer){

    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    int NodeOwnerKey[6];
    int NodeOwnerIdx[6];
    for(int i=offset;i<right;i+=stride){
        int nowDepth = DepthBuffer[i];
        float halfWidth = 1.0f/(1<<(nowDepth+1));
        float Width = 1.0f/(1<<nowDepth);
        float Widthsq = Width * Width;
        Point3D<float> neighCenter[27];
        int neigh[27];
#pragma unroll
        for(int k=0;k<27;++k){
            neigh[k]=NodeArray[i].neighs[k];
            if(neigh[k] != -1){
                neighCenter[k]=CenterBuffer[neigh[k]];
            }
        }
        const Point3D<float> &nodeCenter = neighCenter[13];

        Point3D<float> faceCenterPos[6];
        int orientation;
        int off;
        int multi;
        for(int j=0;j<6;++j){
            orientation = j>>1;
            off = j&1;
            multi = (2*off)-1;
            faceCenterPos[j].coords[0] = nodeCenter.coords[0];
            faceCenterPos[j].coords[1] = nodeCenter.coords[1];
            faceCenterPos[j].coords[2] = nodeCenter.coords[2];
            faceCenterPos[j].coords[orientation] += multi * halfWidth;
        }

        for(int j=0;j<6;++j){
            NodeOwnerKey[j]=0x7fffffff;
        }
        for(int j=0;j<6;++j){
            for(int k=0;k<27;++k){
                if(neigh[k] != -1 && SquareDistance(faceCenterPos[j],neighCenter[k]) < Widthsq){
                    int neighKey = NodeArray[neigh[k]].key;
                    if(NodeOwnerKey[j]>neighKey){
                        NodeOwnerKey[j]=neighKey;
                        NodeOwnerIdx[j]=neigh[k];
                    }
                }
            }
        }

        int parent = NodeArray[i].parent;
        int sonKey = ( NodeArray[i].key >> (3 * (maxDepth-nowDepth)) ) & 7;
        for(int j=0;j<6;++j){
            if(NodeOwnerIdx[j] == i){
                int faceIdx = 6 * (i-left) + j;
                preFaceArray[faceIdx].ownerNodeIdx = i;
                preFaceArray[faceIdx].faceKind = j;
                if(parent == -1){
                    preFaceArray[faceIdx].hasParentFace = -1;
                }else{
                    if(parentFaceKind[sonKey][j] != -1){
                        preFaceArray[faceIdx].hasParentFace = 1;
                    }else{
                        preFaceArray[faceIdx].hasParentFace = -1;
                    }
                }
            }
        }

    }
}

struct validFace{
    __device__ bool operator()(const FaceNode &x){
        return x.ownerNodeIdx >= 0;
    }
};

__global__ void maintainFaceNodePointer(FaceNode *FaceArray,int FaceArray_sz,
                                        OctNode *NodeArray,
                                        int *DepthBuffer,Point3D<float> *CenterBuffer){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<FaceArray_sz;i+=stride){
        int owner = FaceArray[i].ownerNodeIdx;
        int depth = DepthBuffer[owner];
        float halfWidth = 1.0f/(1<<(depth+1));
        float Width = 1.0f/(1<<depth);
        float Widthsq = Width * Width;

        Point3D<float> neighCenter[27];
        int neigh[27];
        for(int k=0;k<27;++k){
            neigh[k]=NodeArray[owner].neighs[k];
            if(neigh[k] != -1){
                neighCenter[k]=CenterBuffer[neigh[k]];
            }
        }

        const Point3D<float> &nodeCenter = neighCenter[13];
        Point3D<float> faceCenterPos;
        int kind = FaceArray[i].faceKind;
        int orientation = kind >> 1;
        int off = kind & 1;
        int multi = (2*off)-1;

        faceCenterPos.coords[0] = nodeCenter.coords[0];
        faceCenterPos.coords[1] = nodeCenter.coords[1];
        faceCenterPos.coords[2] = nodeCenter.coords[2];
        faceCenterPos.coords[orientation] += multi *halfWidth;

        int cnt=0;
        for(int k=0;k<27;++k){
            if(neigh[k] != -1 && SquareDistance(faceCenterPos,neighCenter[k]) < Widthsq){
                FaceArray[i].nodes[cnt] = neigh[k];
                ++cnt;
                int idx = orientation << 1;
                if(neighCenter[k].coords[orientation]-faceCenterPos.coords[orientation]<0)
                    ++idx;

                NodeArray[neigh[k]].faces[idx] = i+1;
            }
        }
    }
}

__global__ void ProcessLeafNodesAtOtherDepth(OctNode *NodeArray,int left,int right,
                                             VertexNode *VertexArray, float *vvalue,
                                             int *hasSurfaceIntersection)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    for(int i=offset;i<right;i+=stride){
        OctNode nowNode = NodeArray[i];
        int hasTri=0;
        int sign = (vvalue[nowNode.vertices[0]-1] < 0 )? -1:1;
        for(int j=1;j<8;++j){
            if(sign * vvalue[nowNode.vertices[j]-1] < 0) {
                hasTri=1;
                break;
            }
        }
        NodeArray[i].hasTriangle=hasTri;

        int hasIntersection=0;
        for(int j=0;j<6;++j){
            if(hasSurfaceIntersection[nowNode.faces[j]-1]){
                hasIntersection=1;
                break;
            }
        }
        NodeArray[i].hasIntersection=hasIntersection;
    }
}

struct ifSubdivide{
    __device__ bool operator()(const OctNode &x){
        return (x.children[0] == -1) && (x.hasTriangle || x.hasIntersection);
    }
};

__global__ void precomputeSubdivideDepth(OctNode *SubdivideNode,int SubdivideNum,
                                         int *DepthBuffer,
                                         int *SubdivideDepthBuffer,
                                         int *SubdivideDepthNum)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<SubdivideNum;i+=stride){
        int nowNodeId = SubdivideNode[i].neighs[13];
        int depth = DepthBuffer[nowNodeId];
        SubdivideDepthBuffer[i] = depth;
        SubdivideDepthNum[i + depth * SubdivideNum] = 1;
    }
}

// correct, but the local memory doesn't support this function to run successfully
// deprecated
__global__ void initFixedDepthNums(OctNode *SubdivideNode,int left,int right,
                                   int *SubdivideDepthBuffer,
                                   int *fixedDepthNums)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset += left;
    for(int i=offset;i<right;i+=stride) {
        int nodeNum=1;
        for(int depth=SubdivideDepthBuffer[i];depth <= maxDepth;++depth){
            fixedDepthNums[ (depth-1) * (right-left) + i - left] = nodeNum;
//            nodeNum *= 8;
            nodeNum <<= 3;
        }
    }
}

// correct, but the local memory doesn't support this function to run successfully
// deprecated
__global__ void wholeRebuildArray(OctNode *SubdivideNode,int left,int right,
                                  OctNode *NodeArray,int NodeArray_sz,
                                  int *SubdivideDepthBuffer,
                                  int *depthNodeAddress_d,int *fixedDepthAddress,
                                  EasyOctNode *RebuildArray,
                                  int *RebuildDepthBuffer,Point3D<float> *RebuildCenterBuffer,
                                  int *ReplaceNodeId,int *IsRoot)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset += left;
    int depthNodeAddress[maxDepth+1];
    for(int i=0;i<=maxDepth;++i){
        depthNodeAddress[i]=depthNodeAddress_d[i];
    }
    int finerSubdivideNum = right - left;
    for(int i=offset;i<right;i+=stride){
        int nowDepth = SubdivideDepthBuffer[i];
        int relativeId = i - left;
        int fixedDepthOffset = fixedDepthAddress[(nowDepth-1) * finerSubdivideNum + relativeId];
        int nowIdx = depthNodeAddress[nowDepth] + fixedDepthOffset;
        OctNode rootNode = SubdivideNode[i];
        int replacedId = rootNode.neighs[13];
        rootNode.neighs[13] = NodeArray_sz + nowIdx;
        RebuildArray[nowIdx] = rootNode;

        ReplaceNodeId[nowIdx] = replacedId;

        RebuildDepthBuffer[nowIdx] = nowDepth;

        IsRoot[nowIdx] = 1;
        Point3D<float> thisNodeCenter;
        getNodeCenterAllDepth(rootNode.key,thisNodeCenter,nowDepth);
        RebuildCenterBuffer[nowIdx] = thisNodeCenter;

        int sonKey = ( rootNode.key >> (3 * (maxDepth-nowDepth)) ) & 7;
        NodeArray[rootNode.parent].children[sonKey] = NodeArray_sz + nowIdx;
        int parentNodeIdx;
        int childrenNums=8;
        while(nowDepth < maxDepth) {
            ++nowDepth;
            fixedDepthOffset = fixedDepthAddress[(nowDepth - 1) * finerSubdivideNum + relativeId];
            nowIdx = depthNodeAddress[nowDepth] + fixedDepthOffset;
            for (int j = 0; j < childrenNums; j+=8) {
                int fatherFixedDepthOffset = fixedDepthAddress[(nowDepth-2) * finerSubdivideNum + relativeId];
                parentNodeIdx = depthNodeAddress[nowDepth-1] + fatherFixedDepthOffset + j/8;
                int parentGlobalIdx=RebuildArray[parentNodeIdx].neighs[13];
                int parentKey=RebuildArray[parentNodeIdx].key;
                for(int k=0;k<8;++k){
                    int thisRoundIdx = nowIdx + j + k;
                    int nowKey = parentKey | (k << (3 * (maxDepth - nowDepth)));
                    RebuildArray[thisRoundIdx].parent = parentGlobalIdx;
                    RebuildArray[thisRoundIdx].key = nowKey;
                    RebuildArray[thisRoundIdx].neighs[13] = NodeArray_sz + thisRoundIdx;

                    ReplaceNodeId[thisRoundIdx] = replacedId;

                    RebuildDepthBuffer[thisRoundIdx] = nowDepth;

                    getNodeCenterAllDepth(nowKey,thisNodeCenter,nowDepth);
                    RebuildCenterBuffer[thisRoundIdx] = thisNodeCenter;

                    RebuildArray[parentNodeIdx].children[k] = NodeArray_sz + thisRoundIdx;
                }
            }
            childrenNums<<=3;
        }
    }
}

__device__ int getSubdivideDepth(const int &rootDepth,const int &idx){
    int up = idx*7+1;
    int base = 8;
    int relativeDepth = 0;
    while(base <= up){
        ++relativeDepth;
        base<<=3;
    }
    return rootDepth + relativeDepth;
}

__global__ void singleRebuildArray(int NodeArray_sz,
                                   OctNode *SubdivideNode,int *SubdivideDepthBuffer,int iterRound,
                                   EasyOctNode *SubdivideArray,int SubdivideArray_sz,
                                   int *SubdivideArrayDepthBuffer,Point3D<float> *SubdivideArrayCenterBuffer)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int rootId = SubdivideNode[iterRound].neighs[13];
    int rootDepth = SubdivideDepthBuffer[iterRound];
    int rootKey = SubdivideNode[iterRound].key;
    for(int i=offset;i<SubdivideArray_sz;i+=stride){
        int thisNodeDepth = getSubdivideDepth(rootDepth,i);
        int relativeDepth = thisNodeDepth - rootDepth;
        int idxOffset = i - (qpow(8,relativeDepth)-1)/7;

        if(thisNodeDepth < maxDepth){
            int nextDepthAddress = (qpow(8,relativeDepth+1)-1)/7;
            for(int k=0;k<8;++k){
                SubdivideArray[i].children[k] = NodeArray_sz + nextDepthAddress + (idxOffset << 3) + k;
            }
        }else{
            for(int k=0;k<8;++k){
                SubdivideArray[i].children[k] = -1;
            }
        }

        if(i != 0){
            int parentDepthAddress = (qpow(8,relativeDepth-1)-1)/7;
            SubdivideArray[i].parent = NodeArray_sz + parentDepthAddress + (idxOffset >> 3);
        }

        int thisKey = rootKey;
        thisKey |= (idxOffset) << (3 * (maxDepth-thisNodeDepth));
        SubdivideArray[i].key = thisKey;

        SubdivideArrayDepthBuffer[i] = thisNodeDepth;
        Point3D<float> thisNodeCenter;
        getNodeCenterAllDepth(thisKey,thisNodeCenter,thisNodeDepth);
        SubdivideArrayCenterBuffer[i]=thisNodeCenter;
    }
}

__global__ void computeRebuildNeighbor(OctNode *SubdivideArray,int left,int right,
                                       OctNode *NodeArray,int NodeArray_sz,
                                       int depthD)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    for(int i=offset;i<right;i+=stride){
        for(int j=0;j<27;++j){
            int sonKey = ( SubdivideArray[i].key >> (3 * (maxDepth-depthD)) ) & 7;
            int parentIdx = SubdivideArray[i].parent;
            int neighParent;
            if(parentIdx < NodeArray_sz){
                neighParent = NodeArray[ parentIdx ].neighs[LUTparent[sonKey][j]];
            }else{
                neighParent = SubdivideArray[ parentIdx - NodeArray_sz ].neighs[LUTparent[sonKey][j]];
            }
            if(neighParent != -1){
                if(neighParent < NodeArray_sz) {
                    SubdivideArray[i].neighs[j] = NodeArray[neighParent].children[LUTchild[sonKey][j]];
                }else{
                    SubdivideArray[i].neighs[j] = SubdivideArray[neighParent - NodeArray_sz ].children[LUTchild[sonKey][j]];
                }
            }else{
                SubdivideArray[i].neighs[j]= -1;
            }
        }
    }
}

__global__ void computeRebuildNeighbor(EasyOctNode *SubdivideArray,int left,int right,
                                       OctNode *NodeArray,int NodeArray_sz,
                                       int depthD)
{
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    offset+=left;
    for(int i=offset;i<right;i+=stride){
        for(int j=0;j<27;++j){
            int sonKey = ( SubdivideArray[i].key >> (3 * (maxDepth-depthD)) ) & 7;
            int parentIdx = SubdivideArray[i].parent;
            int neighParent;
            if(parentIdx < NodeArray_sz){
                neighParent = NodeArray[ parentIdx ].neighs[LUTparent[sonKey][j]];
            }else{
                neighParent = SubdivideArray[ parentIdx - NodeArray_sz ].neighs[LUTparent[sonKey][j]];
            }
            if(neighParent != -1){
                if(neighParent < NodeArray_sz) {
                    SubdivideArray[i].neighs[j] = NodeArray[neighParent].children[LUTchild[sonKey][j]];
                }else{
                    SubdivideArray[i].neighs[j] = SubdivideArray[neighParent - NodeArray_sz ].children[LUTchild[sonKey][j]];
                }
            }else{
                SubdivideArray[i].neighs[j]= -1;
            }
        }
    }
}


__host__ void insertTriangle(Point3D<float> *VertexBuffer,const int &allVexNums,
                             int *TriangleBuffer,const int &allTriNums,
                             CoredVectorMeshData &mesh)
{
    int previousVertex = mesh.inCorePoints.size();
    for(int i=0;i<allVexNums;++i){
        if(abs(VertexBuffer[i].coords[0])<EPSILON)
            printf("error\n");
        mesh.inCorePoints.push_back(VertexBuffer[i]);
    }
    int inCoreFlag=0;
    for(int i=0;i<3;++i){
        inCoreFlag|=CoredMeshData::IN_CORE_FLAG[i];
    }
    for(int i=0;i<allTriNums;++i){
        TriangleIndex tri;
        for(int j=0;j<3;++j) {
            tri.idx[j] = TriangleBuffer[3*i+j] + previousVertex;
            if(tri.idx[j]<0 || tri.idx[j]>=allVexNums + previousVertex){
                printf("%d %d\n",tri.idx[j]-previousVertex,allVexNums);
                printf("tri error\n");
            }
        }
        mesh.addTriangle(tri,inCoreFlag);
    }
}

int main() {
//    char fileName[]="/home/davidxu/horse.npts";
//    char outName[]="/home/davidxu/horse.ply";

    char fileName[]="/home/davidxu/bunny.points.ply";
    char outName[]="/home/davidxu/bunny.ply";

//    char fileName[]="/home/davidxu/eagle.points.ply";
//    char outName[]="/home/davidxu/eagle.ply";

    int NodeArrayCount_h[maxDepth_h+1];
    int BaseAddressArray[maxDepth_h+1];

    Point3D<float> *samplePoints_d=NULL, *sampleNormals_d=NULL;
    int *PointToNodeArrayD;
    OctNode *NodeArray;
    int count=0;
    int NodeArray_sz=0;
    Point3D<float> center;
    float scale;

    double project_st=cpuSecond();

    // the number of nodes at maxDepth is very large, some maintaining of their info is time-consuming
    pipelineBuildNodeArray(fileName,center,scale,count,NodeArray_sz,
                           NodeArrayCount_h,BaseAddressArray,
                           samplePoints_d,sampleNormals_d,PointToNodeArrayD,NodeArray );

    printf("NodeArray_sz:%d\n",NodeArray_sz);

//    outputDeviceArray<<<1,1>>>(PointToNodeArrayD,20);
//    cudaDeviceSynchronize();

    int *BaseAddressArray_d=NULL;
    CHECK(cudaMalloc((int **)&BaseAddressArray_d,sizeof(int)*(maxDepth_h+1) ));
    CHECK(cudaMemcpy(BaseAddressArray_d,BaseAddressArray,sizeof(int)*(maxDepth_h+1),cudaMemcpyHostToDevice));
//    for(int i=0;i<=maxDepth_h;++i){
//        printf("%d %d\n",NodeArrayCount_h[i],BaseAddressArray[i]);
//    }


//    OctNode *a=(OctNode *)malloc(sizeof(OctNode)*(BaseAddressArray[maxDepth_h]+NodeArrayCount_h[maxDepth_h]));
//    cudaMemcpy(a,NodeArray,sizeof(OctNode)*(BaseAddressArray[maxDepth_h]+NodeArrayCount_h[maxDepth_h]),cudaMemcpyDeviceToHost);
//    for(int i=BaseAddressArray[1];i<BaseAddressArray[3];++i){
//        std::cout<<std::bitset<32>(a[i].key)<<" pidx:"<<a[i].pidx<<" pnum:"<<a[i].pnum<<std::endl;
//        int idx[3];
//        getFunctionIdxOfNode(a[i].key, getDepth(i,BaseAddressArray),idx);
//        std::cout<<getDepth(i,BaseAddressArray)<<std::endl;
//        for(int j=0;j<3;++j){
//            printf("idx[%d]:%d ",j,idx[j]);
//        }
//        puts("");
//    }

    // ----------------------------------------------------

    double cpu_st=cpuSecond();

    PPolynomial<convTimes> ReconstructionFunction = PPolynomial<convTimes>::GaussianApproximation();
    FunctionData<convTimes,double> fData;
    fData.set(maxDepth_h,ReconstructionFunction,normalize,0);
    //  precomputed inner product table may can be optimized to GPU parallel
    fData.setDotTables(fData.DOT_FLAG | fData.D_DOT_FLAG | fData.D2_DOT_FLAG);
    PPolynomial<convTimes> &F=ReconstructionFunction;
    switch(normalize){
        case 2:
            F=F/sqrt((F*F).integral(F.polys[0].start,F.polys[F.polyCount-1].start));
            break;
        case 1:
            F=F/F.integral(F.polys[0].start,F.polys[F.polyCount-1].start);
            break;
        default:
            F=F/F(0);
    }

    long long nByte = 1ll * sizeof(double) * fData.res * fData.res;
    double *dot_F_F=NULL;
    CHECK(cudaMalloc((double **)&dot_F_F,nByte));
    CHECK(cudaMemcpy(dot_F_F,fData.dotTable,nByte,cudaMemcpyHostToDevice));

    double *dot_F_DF=NULL;
    CHECK(cudaMalloc((double **)&dot_F_DF,nByte));
    CHECK(cudaMemcpy(dot_F_DF,fData.dDotTable,nByte,cudaMemcpyHostToDevice));

    double *dot_F_D2F=NULL;
    CHECK(cudaMalloc((double **)&dot_F_D2F,nByte));
    CHECK(cudaMemcpy(dot_F_D2F,fData.d2DotTable,nByte,cudaMemcpyHostToDevice));

    fData.clearDotTables(fData.DOT_FLAG | fData.D_DOT_FLAG | fData.D2_DOT_FLAG);

    ConfirmedPPolynomial<convTimes+1,convTimes+2> baseFunctions_h[fData.res];
    for(int i=0;i<fData.res;++i){
        baseFunctions_h[i]=fData.baseFunctions[i];
    }

    ConfirmedPPolynomial<convTimes+1,convTimes+2> *baseFunctions_d=NULL;
    nByte = 1ll * sizeof(ConfirmedPPolynomial<convTimes+1,convTimes+2>) * fData.res;
    CHECK(cudaMalloc((ConfirmedPPolynomial<convTimes+1,convTimes+2>**)&baseFunctions_d,nByte));
    CHECK(cudaMemcpy(baseFunctions_d,baseFunctions_h,nByte,cudaMemcpyHostToDevice));

    double cpu_ed=cpuSecond();
    printf("CPU generate precomputed inner product table takes:%lfs\n",cpu_ed-cpu_st);

    // ----------------------------------------------------

    ConfirmedPPolynomial<convTimes,convTimes+2> BaseFunctionMaxDepth(ReconstructionFunction.scale(1.0/(1<<maxDepth_h)));
    nByte = 1ll * sizeof(BaseFunctionMaxDepth);
    ConfirmedPPolynomial<convTimes,convTimes+2> *BaseFunctionMaxDepth_d= NULL;
    CHECK(cudaMalloc((ConfirmedPPolynomial<convTimes,convTimes+2>**)&BaseFunctionMaxDepth_d,nByte));
    CHECK(cudaMemcpy(BaseFunctionMaxDepth_d,&BaseFunctionMaxDepth,nByte,cudaMemcpyHostToDevice));

    int NodeDNum=NodeArrayCount_h[maxDepth_h];

    Point3D<float> *VectorField=NULL;
    nByte = 1ll * sizeof(Point3D<float>) * NodeArrayCount_h[maxDepth_h];
    CHECK(cudaMalloc((Point3D<float> **)&VectorField,nByte));
    CHECK(cudaMemset(VectorField,0,nByte));

    double st=cpuSecond();
    dim3 grid=(32,32);
    dim3 block(32,32);
    computeVectorField<<<grid,block>>>(BaseFunctionMaxDepth_d,samplePoints_d,sampleNormals_d,
                                       NodeArray,BaseAddressArray[maxDepth_h],NodeArray_sz,VectorField);
    cudaDeviceSynchronize();

//    outputDeviceArray<<<1,1>>>(VectorField,200);
//    cudaDeviceSynchronize();

    double mid1=cpuSecond();
    printf("Compute Vector Field takes:%lfs\n",mid1-st);

    // ----------------------------------------------------

    float *Divergence=NULL;
    nByte = 1ll * sizeof(float) * NodeArray_sz;
    CHECK(cudaMallocManaged((float **)&Divergence,nByte));
    CHECK(cudaMemset(Divergence,0,nByte));

    int *EncodedNodeIdxInFunction=NULL;
    nByte = 1ll * sizeof(int) * NodeArray_sz;
    CHECK(cudaMalloc((int **)&EncodedNodeIdxInFunction, nByte));
    precomputeEncodedFunctionIdxOfNode<<<grid,block>>>(BaseAddressArray_d,
                                                       NodeArray, NodeArray_sz,
                                                       EncodedNodeIdxInFunction);
    cudaDeviceSynchronize();
    double mid2=cpuSecond();
    printf("Precompute Function index of node takes:%lfs\n",mid2-mid1);

    // memory access is very slow, maybe optimize it by setting faster memory.
    printf("left:%d,right:%d\n",BaseAddressArray[5],NodeArray_sz);
    computeEncodedFinerNodesDivergence<<<grid,block>>>(BaseAddressArray_d, EncodedNodeIdxInFunction,
                                                       NodeArray, BaseAddressArray[5],BaseAddressArray[maxDepth_h]+NodeArrayCount_h[maxDepth_h],
                                                       VectorField, dot_F_DF,
                                                       Divergence);
    cudaDeviceSynchronize();

//    float *Divergence_h=(float *)malloc(sizeof(float)*NodeArray_sz);
//    cudaMemcpy(Divergence_h,Divergence,sizeof(float)*NodeArray_sz,cudaMemcpyDeviceToHost);
//    for(int i=BaseAddressArray[5];i<BaseAddressArray[6];++i){
//        printf("%f\n",Divergence[i]);
//    }

    double mid3=cpuSecond();
    printf("Compute finer depth nodes' divergence takes:%lfs\n",mid3-mid2);

    // ----------------------------------------------------

    // maybe can be optimized by running all nodes at the same time.
//    nByte = 1ll * sizeof(float) * NodeDNum;
    for(int i=4;i>=0;--i){
        for(int j=BaseAddressArray[i];j<BaseAddressArray[i+1];++j){
            int *coverNums=NULL;
            CHECK(cudaMalloc((int**)&coverNums,sizeof(int) * 28));
            computeCoverNums<<<1,1>>>(NodeArray,j,coverNums);
            cudaDeviceSynchronize();
            int coverNums_h[28];
            CHECK(cudaMemcpy(coverNums_h,coverNums,sizeof(int) * 28,cudaMemcpyDeviceToHost));
//            printf("%d,%d\n",j,coverNums_h);

            float *divg=NULL;
            nByte = 1ll * sizeof(float)*coverNums_h[27];
            CHECK(cudaMalloc((float**)&divg,nByte));
            CHECK(cudaMemset(divg,0,nByte));

            int *DIdxArray=NULL;
            nByte = 1ll * sizeof(int)*coverNums_h[27];
            CHECK(cudaMalloc((int**)&DIdxArray,nByte));
            CHECK(cudaMemset(DIdxArray,0,nByte));

            generateDIdxArray<<<grid,block>>>(NodeArray,j,coverNums,DIdxArray);
            cudaDeviceSynchronize();

            computeEncodedCoarserNodesDivergence<<<grid,block>>>(DIdxArray, coverNums_h[27], BaseAddressArray_d,
                                                                 EncodedNodeIdxInFunction,
                                                                 VectorField, dot_F_DF,
                                                                 j, divg);
            cudaDeviceSynchronize();
            thrust::device_ptr<float> divg_ptr=thrust::device_pointer_cast<float>(divg);
            float val=thrust::reduce(divg_ptr,divg_ptr+coverNums_h[27]);
            cudaDeviceSynchronize();

            CHECK(cudaMemcpy(Divergence+j,&val,sizeof(float),cudaMemcpyHostToDevice));

            cudaFree(DIdxArray);
            cudaFree(divg);
        }
    }
    cudaFree(VectorField);

    double mid4=cpuSecond();
    printf("Compute coarser depth nodes' divergence takes:%lfs\n",mid4-mid3);

    // ----------------------------------------------------

    // d_x is the Solution
    float *d_x=NULL;
    LaplacianIteration(BaseAddressArray,NodeArrayCount_h,4,
                       EncodedNodeIdxInFunction,NodeArray,Divergence,
                       NodeArray_sz,
                       dot_F_F,dot_F_D2F,
                       d_x);
    cudaFree(Divergence);

    double mid5=cpuSecond();
    printf("GPU Laplacian Iteration takes:%lfs\n",mid5-mid4);

    // ----------------------------------------------------

    float *pointValue=NULL;
    nByte = 1ll * sizeof(float)*count;
    CHECK(cudaMalloc((float**)&pointValue,nByte));
    CHECK(cudaMemset(pointValue,0,nByte));

    grid=(32,32);
    block=(32,32);
    calculatePointsImplicitFunctionValue<<<grid,grid>>>(samplePoints_d,PointToNodeArrayD,count,BaseAddressArray[maxDepth_h],
                                                         NodeArray,d_x,
                                                         EncodedNodeIdxInFunction,baseFunctions_d,
                                                         pointValue);
    cudaDeviceSynchronize();

    thrust::device_ptr<float> pointValue_ptr=thrust::device_pointer_cast<float>(pointValue);
    float isoValue=thrust::reduce(pointValue_ptr,pointValue_ptr+count);
    cudaDeviceSynchronize();
    isoValue/=count;

    double mid6 = cpuSecond();
    printf("isoValue:%f\nGPU calculate isoValue takes:%lfs\n",isoValue,mid6-mid5);

    // ----------------------------------------------------

    // pre-compute the center of node ?
    int *DepthBuffer=NULL;
    nByte = 1ll * sizeof(int) * NodeArray_sz;
    CHECK(cudaMalloc((int**)&DepthBuffer,nByte));

    Point3D<float> *CenterBuffer=NULL;
    nByte = 1ll * sizeof(Point3D<float>) * NodeArray_sz;
    CHECK(cudaMallocManaged((Point3D<float>**)&CenterBuffer,nByte));

    precomputeDepthAndCenter<<<grid,block>>>(BaseAddressArray_d,NodeArray,NodeArray_sz,
                                             DepthBuffer,CenterBuffer);
    cudaDeviceSynchronize();



    // generate all depth vertex
    VertexNode *preVertexArray=NULL;
    nByte = 1ll * sizeof(VertexNode) * 8 * NodeArray_sz;
    CHECK(cudaMalloc((VertexNode**)&preVertexArray,nByte));
    CHECK(cudaMemset(preVertexArray,0,nByte));
    grid=(32,32);
    block=(32,32);
    initVertexOwner<<<grid,block>>>(NodeArray,0,NodeArray_sz,
                                    preVertexArray,
                                    DepthBuffer,CenterBuffer);
    cudaDeviceSynchronize();

    VertexNode *VertexArray=NULL;
//    nByte = 1ll * sizeof(VertexNode) * 8 * NodeArray_sz;
    CHECK(cudaMalloc((VertexNode**)&VertexArray,nByte));
    CHECK(cudaMemset(VertexArray,0,nByte));
    thrust::device_ptr<VertexNode> preVertexArray_ptr=thrust::device_pointer_cast<VertexNode>(preVertexArray);
    thrust::device_ptr<VertexNode> VertexArray_ptr=thrust::device_pointer_cast<VertexNode>(VertexArray);
    thrust::device_ptr<VertexNode> VertexArray_end=thrust::copy_if(preVertexArray_ptr,preVertexArray_ptr+8*NodeArray_sz,VertexArray_ptr,validVertex());
    cudaDeviceSynchronize();

    cudaFree(preVertexArray);

    int VertexArray_sz=VertexArray_end-VertexArray_ptr;

    maintainVertexNodePointerNonAtomic<<<grid,block>>>(VertexArray,VertexArray_sz,
                                                       NodeArray,
                                                       DepthBuffer,CenterBuffer);
    cudaDeviceSynchronize();

//    OctNode *a=(OctNode *)malloc(sizeof(OctNode)*NodeArray_sz);
//    cudaMemcpy(a,NodeArray,sizeof(OctNode)*(BaseAddressArray[maxDepth_h]+NodeArrayCount_h[maxDepth_h]),cudaMemcpyDeviceToHost);
//    for(int j=maxDepth_h;j<=maxDepth_h;++j) {
//        int all=0;
//        for (int i = BaseAddressArray[j]; i < BaseAddressArray[j]+100; ++i) {
////            if(a[i].pnum==0) continue;
//            all+=a[i].dnum;
//            std::cout << i << " " <<std::bitset<32>(a[i].key) << " pidx:" << a[i].pidx << " pnum:" << a[i].pnum << " parent:"
//                      << a[i].parent << " didx:"<< a[i].didx << " dnum:" << a[i].dnum << std::endl;
//            for(int k=0;k<8;++k){
//                printf("children[%d]:%d ",k,a[i].children[k]);
//            }
//            puts("");
//            for(int k=0;k<27;++k){
//                printf("neigh[%d]:%d ",k,a[i].neighs[k]);
//            }
//            puts("");
//            for(int k=0;k<8;++k){
//                printf("vertices[%d]:%d ",k,a[i].vertices[k]);
//            }
//            puts("");
//        }
//        printf("allD:%d\n",all);
//        std::cout<<std::endl;
//    }

    double mid7=cpuSecond();
    printf("VertexArray_sz:%d\nGPU build VertexArray takes:%lfs\n",VertexArray_sz,mid7-mid6);

    // ----------------------------------------------------

    // generate the edge at maxDepth
    EdgeNode *preEdgeArray=NULL;
    nByte = 1ll * sizeof(EdgeNode) * 12 *NodeDNum;
    CHECK(cudaMalloc((EdgeNode**)&preEdgeArray,nByte));
    CHECK(cudaMemset(preEdgeArray,0,nByte));

    initEdgeArray<<<grid,block>>>(NodeArray,BaseAddressArray[maxDepth_h],NodeArray_sz,
                                  preEdgeArray,
                                  DepthBuffer,CenterBuffer);
    cudaDeviceSynchronize();

    EdgeNode *EdgeArray=NULL;
//    nByte = 1ll * sizeof(VertexNode) * 12 * NodeDNum;
    CHECK(cudaMalloc((EdgeNode**)&EdgeArray,nByte));
    CHECK(cudaMemset(EdgeArray,0,nByte));
    thrust::device_ptr<EdgeNode> preEdgeArray_ptr=thrust::device_pointer_cast<EdgeNode>(preEdgeArray);
    thrust::device_ptr<EdgeNode> EdgeArray_ptr=thrust::device_pointer_cast<EdgeNode>(EdgeArray);
    thrust::device_ptr<EdgeNode> EdgeArray_end=thrust::copy_if(preEdgeArray_ptr,preEdgeArray_ptr+12*NodeDNum,EdgeArray_ptr,validEdge());
    cudaDeviceSynchronize();

    cudaFree(preEdgeArray);

    int EdgeArray_sz=EdgeArray_end-EdgeArray_ptr;

    maintainEdgeNodePointer<<<grid,block>>>(EdgeArray,EdgeArray_sz,
                                            NodeArray,
                                            DepthBuffer,CenterBuffer);
    cudaDeviceSynchronize();

    double mid8=cpuSecond();
    printf("EdgeArray_sz:%d\nGPU build EdgeArray takes:%lfs\n",EdgeArray_sz,mid8-mid7);

    // ----------------------------------------------------


    // generate all depth face
    FaceNode *preFaceArray=NULL;
    nByte = 1ll * sizeof(FaceNode) * 6 * NodeArray_sz;
    CHECK(cudaMalloc((FaceNode**)&preFaceArray,nByte));
    CHECK(cudaMemset(preFaceArray,-1,nByte));

    initFaceArray<<<grid,block>>>(NodeArray,0,NodeArray_sz,preFaceArray,DepthBuffer,CenterBuffer);
    cudaDeviceSynchronize();

    FaceNode *FaceArray = NULL;
//    nByte = 1ll * sizeof(FaceNode) * 6 * NodeArray_sz;
    CHECK(cudaMalloc((FaceNode**)&FaceArray,nByte));
    CHECK(cudaMemset(FaceArray,-1,nByte));
    thrust::device_ptr<FaceNode> preFaceArray_ptr=thrust::device_pointer_cast<FaceNode>(preFaceArray);
    thrust::device_ptr<FaceNode> FaceArray_ptr=thrust::device_pointer_cast<FaceNode>(FaceArray);
    thrust::device_ptr<FaceNode> FaceArray_end=thrust::copy_if(preFaceArray_ptr,preFaceArray_ptr+6*NodeArray_sz,FaceArray_ptr,validFace());
    cudaDeviceSynchronize();

    int FaceArray_sz = FaceArray_end - FaceArray_ptr;

    cudaFree(preFaceArray);

    maintainFaceNodePointer<<<grid,block>>>(FaceArray,FaceArray_sz,
                                            NodeArray,
                                            DepthBuffer,CenterBuffer);
    cudaDeviceSynchronize();

    double mid_insert=cpuSecond();
    printf("FaceArray_sz:%d\nGPU build FaceArray takes:%lfs\n",FaceArray_sz,mid_insert-mid8);

    int *hasSurfaceIntersection=NULL;
    nByte = 1ll * sizeof(int) * FaceArray_sz;
    CHECK(cudaMalloc((int**)&hasSurfaceIntersection,nByte));
    CHECK(cudaMemset(hasSurfaceIntersection,0,nByte));


    // ----------------------------------------------------

    // Step 1: compute implicit function values for octree vertices
    float *vvalue = NULL;
    nByte = 1ll * sizeof(float) * VertexArray_sz;
    CHECK(cudaMalloc((float**)&vvalue,nByte));
    CHECK(cudaMemset(vvalue,0,nByte));

    computeVertexImplicitFunctionValue<<<grid,block>>>(VertexArray,VertexArray_sz,
                                                       NodeArray,d_x,
                                                       EncodedNodeIdxInFunction,baseFunctions_d,
                                                       vvalue,isoValue);
    cudaDeviceSynchronize();

    double mid9=cpuSecond();
    printf("Compute vertex implicit function value takes:%lfs\n",mid9-mid_insert);

    // Step 2: compute vertex number and address
    int *vexNums=NULL;
    nByte = 1ll * sizeof(int) * EdgeArray_sz;
    CHECK(cudaMalloc((int**)&vexNums,nByte));
    CHECK(cudaMemset(vexNums,0,nByte));

    generateVexNums<<<grid,block>>>(EdgeArray,EdgeArray_sz,
                                    NodeArray,vvalue,
                                    vexNums);
    cudaDeviceSynchronize();

    int *vexAddress=NULL;
//    nByte = 1ll * sizeof(int) * EdgeArray_sz;
    CHECK(cudaMalloc((int**)&vexAddress,nByte));
    CHECK(cudaMemset(vexAddress,0,nByte));

    thrust::device_ptr<int> vexNums_ptr=thrust::device_pointer_cast<int>(vexNums);
    thrust::device_ptr<int> vexAddress_ptr=thrust::device_pointer_cast<int>(vexAddress);

    thrust::exclusive_scan(vexNums_ptr,vexNums_ptr+EdgeArray_sz,vexAddress_ptr);
    cudaDeviceSynchronize();

    double mid10=cpuSecond();
    printf("Compute vexAddress takes:%lfs\n",mid10-mid9);

    // Step 3: compute triangle number and address
    int *triNums=NULL;
    nByte = 1ll * sizeof(int) * NodeDNum;
    CHECK(cudaMalloc((int**)&triNums,nByte));
    CHECK(cudaMemset(triNums,0,nByte));

    int *cubeCatagory=NULL;
//    nByte = 1ll * sizeof(int) * NodeDNum;
    CHECK(cudaMalloc((int**)&cubeCatagory,nByte));
    CHECK(cudaMemset(cubeCatagory,0,nByte));

    generateTriNums<<<grid,block>>>(NodeArray,
                                    BaseAddressArray[maxDepth_h],NodeArray_sz,
                                    vvalue,
                                    triNums,cubeCatagory);
    cudaDeviceSynchronize();

    int *triAddress=NULL;
//    nByte = 1ll * sizeof(int) * NodeDNum;
    CHECK(cudaMalloc((int**)&triAddress,nByte));
    CHECK(cudaMemset(triAddress,0,nByte));

    thrust::device_ptr<int> triNums_ptr=thrust::device_pointer_cast<int>(triNums);
    thrust::device_ptr<int> triAddress_ptr=thrust::device_pointer_cast<int>(triAddress);

    thrust::exclusive_scan(triNums_ptr,triNums_ptr+NodeDNum,triAddress_ptr);
    cudaDeviceSynchronize();

    double mid11=cpuSecond();
    printf("Compute triAddress takes:%lfs\n",mid11-mid10);


    // Step 4: generate vertices
    int lastVexAddr;
    int lastVexNums;
    CHECK(cudaMemcpy(&lastVexAddr,vexAddress+EdgeArray_sz-1,sizeof(int),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&lastVexNums,vexNums+EdgeArray_sz-1,sizeof(int),cudaMemcpyDeviceToHost));
    int allVexNums = lastVexAddr + lastVexNums;
    Point3D<float> *VertexBuffer=NULL;
    nByte = 1ll * sizeof(Point3D<float>) * allVexNums;
    CHECK(cudaMallocManaged((Point3D<float>**)&VertexBuffer,nByte));
//    CHECK(cudaMemset(VertexBuffer,0,nByte));

    EdgeNode * validEdgeArray = NULL;
    nByte = 1ll * sizeof(EdgeNode) * allVexNums;
    CHECK(cudaMalloc((EdgeNode**)&validEdgeArray,nByte));
    thrust::device_ptr<EdgeNode> validEdgeArray_ptr = thrust::device_pointer_cast<EdgeNode>(validEdgeArray);
    EdgeArray_ptr = thrust::device_pointer_cast<EdgeNode>(EdgeArray);
    thrust::device_ptr<EdgeNode> validEdgeArray_end = thrust::copy_if(EdgeArray_ptr,EdgeArray_ptr+EdgeArray_sz,vexNums_ptr,validEdgeArray_ptr,validVexNums());
    cudaDeviceSynchronize();

    int validEdgeArray_sz = validEdgeArray_end - validEdgeArray_ptr;
    assert(allVexNums == validEdgeArray_sz);

    int *validVexAddress = NULL;
    nByte = 1ll * sizeof(int) * allVexNums;
    CHECK(cudaMalloc((int**)&validVexAddress,nByte));
    thrust::device_ptr<int> validVexAddress_ptr = thrust::device_pointer_cast<int>(validVexAddress);
    thrust::device_ptr<int> validVexAddress_end = thrust::copy_if(vexAddress_ptr,vexAddress_ptr + EdgeArray_sz, vexNums_ptr,validVexAddress_ptr,validVexNums());
    cudaDeviceSynchronize();

    int validVexAddress_sz = validVexAddress_end - validVexAddress_ptr;
    assert(allVexNums == validVexAddress_sz);

    grid = (std::min(allVexNums / 1024 , 32) ,allVexNums/32768);
    generateIntersectionPoint<<<grid,block>>>(validEdgeArray,allVexNums,
                                              VertexArray,NodeArray,
                                              validVexAddress,vvalue,
                                              VertexBuffer);
    cudaDeviceSynchronize();

    grid = (32,32);

    double mid12=cpuSecond();
    printf("Generate interpolate vertices takes:%lfs\n",mid12-mid11);

    // Step 5: generate triangles
    int lastTriAddr;
    int lastTriNums;
    CHECK(cudaMemcpy(&lastTriAddr,triAddress+NodeDNum-1,sizeof(int),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&lastTriNums,triNums+NodeDNum-1,sizeof(int),cudaMemcpyDeviceToHost));
    int allTriNums = lastTriAddr+lastTriNums;

    int *TriangleBuffer=NULL;
    nByte = 1ll * sizeof(int) * 3 * allTriNums;
    CHECK(cudaMallocManaged((int**)&TriangleBuffer,nByte));
//    CHECK(cudaMemset(TriangleBuffer,0,nByte));

    generateTrianglePos<<<grid,block>>>(NodeArray,BaseAddressArray[maxDepth_h],NodeArray_sz,
                                        triNums,cubeCatagory,
                                        vexAddress,
                                        triAddress,TriangleBuffer,
                                        FaceArray,hasSurfaceIntersection);
    cudaDeviceSynchronize();

    double mid13=cpuSecond();
    printf("Process Triangle indices takes:%lfs\n",mid13-mid12);


    CoredVectorMeshData mesh;

    insertTriangle(VertexBuffer,allVexNums,
                   TriangleBuffer,allTriNums,
                   mesh);

    // ----------------------------------------------------

    ProcessLeafNodesAtOtherDepth<<<grid,block>>>(NodeArray,0,BaseAddressArray[maxDepth_h],
                                                 VertexArray,vvalue,
                                                 hasSurfaceIntersection);
    cudaDeviceSynchronize();

    cudaFree(VertexArray);
    cudaFree(EdgeArray);
    cudaFree(FaceArray);
    cudaFree(hasSurfaceIntersection);
//    cudaFree(vvalue);
    cudaFree(vexNums);
    cudaFree(vexAddress);
    cudaFree(triNums);
    cudaFree(cubeCatagory);
    cudaFree(triAddress);
    cudaFree(VertexBuffer);
    cudaFree(TriangleBuffer);
    cudaFree(validEdgeArray);
    cudaFree(validVexAddress);


    // ----------------------------------------------------


    OctNode *SubdivideNode=NULL;
    nByte = 1ll * sizeof(OctNode) * BaseAddressArray[maxDepth_h];
    CHECK(cudaMalloc((OctNode**)&SubdivideNode,nByte));
    CHECK(cudaMemset(SubdivideNode,0,nByte));
    thrust::device_ptr<OctNode> NodeArray_ptr=thrust::device_pointer_cast<OctNode>(NodeArray);
    thrust::device_ptr<OctNode> SubdivideNode_ptr=thrust::device_pointer_cast<OctNode>(SubdivideNode);
    thrust::device_ptr<OctNode> SubdivideNode_end=thrust::copy_if(NodeArray_ptr,NodeArray_ptr+BaseAddressArray[maxDepth_h],SubdivideNode_ptr,ifSubdivide());
    cudaDeviceSynchronize();

    int SubdivideNum = SubdivideNode_end - SubdivideNode_ptr;

    printf("Starting processing the subdivide node\n");

    printf("SubdivideNum:%d\n",SubdivideNum);

    int *SubdivideDepthBuffer=NULL;
    nByte = 1ll * sizeof(int) * SubdivideNum;
    CHECK(cudaMalloc((int**)&SubdivideDepthBuffer,nByte));

    int *SubdivideDepthNum = NULL;
    nByte = 1ll * sizeof(int) * (maxDepth_h+1) * SubdivideNum;
    CHECK(cudaMalloc((int**)&SubdivideDepthNum,nByte));
    CHECK(cudaMemset(SubdivideDepthNum,0,nByte));

    precomputeSubdivideDepth<<<grid,block>>>(SubdivideNode,SubdivideNum,
                                             DepthBuffer,
                                             SubdivideDepthBuffer,
                                             SubdivideDepthNum);
    cudaDeviceSynchronize();

    int SubdivideDepthCount[maxDepth_h+1];

    thrust::device_ptr<int> SubdivideDepthNum_ptr = thrust::device_pointer_cast<int>(SubdivideDepthNum);
    for(int i=0;i<=maxDepth_h;++i){
        SubdivideDepthCount[i]=thrust::reduce(SubdivideDepthNum_ptr + i*SubdivideNum,SubdivideDepthNum_ptr + (i+1)*SubdivideNum);
    }
    cudaDeviceSynchronize();

    cudaFree(SubdivideDepthNum);

    int SubdivideDepthAddress[maxDepth_h+1];
    SubdivideDepthAddress[0]=0;
    for(int i=1;i<=maxDepth_h;++i){
        SubdivideDepthAddress[i]=SubdivideDepthAddress[i-1]+SubdivideDepthCount[i-1];
    }

    int minSubdivideRootDepth;
    CHECK(cudaMemcpy(&minSubdivideRootDepth,SubdivideDepthBuffer,sizeof(int),cudaMemcpyDeviceToHost));
    int maxNodeNums = (qpow(8,(maxDepth_h-minSubdivideRootDepth+1) )-1 )/7;

    EasyOctNode *SubdivideArray=NULL;
    nByte = 1ll * sizeof(EasyOctNode) * maxNodeNums;
    CHECK(cudaMalloc((EasyOctNode**)&SubdivideArray,nByte));

    int *SubdivideArrayDepthBuffer=NULL;
    nByte = 1ll * sizeof(int) * maxNodeNums;
    CHECK(cudaMalloc((int**)&SubdivideArrayDepthBuffer,nByte));

    Point3D<float> *SubdivideArrayCenterBuffer=NULL;
    nByte = 1ll * sizeof(Point3D<float>) * maxNodeNums;
    CHECK(cudaMalloc((Point3D<float>**)&SubdivideArrayCenterBuffer,nByte));



    int finerDepth = 6;
    for(int i=0;i<SubdivideNum;++i){
//        int rootDepth = SubdivideDepthBuffer[i];
        int rootDepth;
        CHECK(cudaMemcpy(&rootDepth,SubdivideDepthBuffer+i,sizeof(int),cudaMemcpyDeviceToHost));
        if(rootDepth >= finerDepth){
            break;
        }
        int SubdivideArray_sz = (qpow(8,(maxDepth_h-rootDepth+1) )-1 )/7;
        int fixedDepthNodeNum[maxDepth+1]={0};
        int nowNodeNum=1;
        fixedDepthNodeNum[rootDepth-1]=0;
        for(int j=rootDepth;j<=maxDepth_h;++j){
            fixedDepthNodeNum[j]=nowNodeNum;
            nowNodeNum<<=3;
        }
        int fixedDepthNodeAddress[maxDepth+1]={0};
        fixedDepthNodeAddress[rootDepth-1]=0;
        for(int j=rootDepth;j<=maxDepth_h;++j){
            fixedDepthNodeAddress[j]=fixedDepthNodeAddress[j-1]+fixedDepthNodeNum[j-1];
//            printf("%d %d\n",fixedDepthNodeNum[j-1],fixedDepthNodeAddress[j]);
        }

        OctNode rootNode;
        CHECK(cudaMemcpy(&rootNode,&SubdivideNode[i],sizeof(OctNode),cudaMemcpyDeviceToHost));
        int rootId;
        rootId = rootNode.neighs[13];
        int rootParent;
        rootParent = rootNode.parent;
        int rootKey = rootNode.key;
        int rootSonKey =( rootKey >> (3 * (maxDepth-rootDepth)) ) & 7 ;

        nByte = 1ll * sizeof(EasyOctNode) * SubdivideArray_sz;
//        CHECK(cudaMalloc((EasyOctNode**)&SubdivideArray,nByte));
        CHECK(cudaMemset(SubdivideArray,0,nByte));

        CHECK(cudaMemcpy(&NodeArray[rootParent].children[rootSonKey],&NodeArray_sz,sizeof(int),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(&SubdivideArray[0].parent,&rootParent,sizeof(int),cudaMemcpyHostToDevice));

//        int *SubdivideArrayDepthBuffer=NULL;
//        nByte = 1ll * sizeof(int) * SubdivideArray_sz;
//        CHECK(cudaMalloc((int**)&SubdivideArrayDepthBuffer,nByte));

//        Point3D<float> *SubdivideArrayCenterBuffer=NULL;
//        nByte = 1ll * sizeof(Point3D<float>) * SubdivideArray_sz;
//        CHECK(cudaMalloc((Point3D<float>**)&SubdivideArrayCenterBuffer,nByte));

        singleRebuildArray<<<grid,block>>>(NodeArray_sz,
                                           SubdivideNode,SubdivideDepthBuffer,i,
                                           SubdivideArray,SubdivideArray_sz,
                                           SubdivideArrayDepthBuffer,SubdivideArrayCenterBuffer);
        cudaDeviceSynchronize();


        for(int j=rootDepth;j<=maxDepth_h;++j) {
            computeRebuildNeighbor<<<grid, block>>>(SubdivideArray,fixedDepthNodeAddress[j],
                                             fixedDepthNodeAddress[j] + fixedDepthNodeNum[j],
                                             NodeArray,NodeArray_sz,
                                             j);
            cudaDeviceSynchronize();

        }

        // ----------------------------------------------------

        // preVertexArray
        VertexNode *SubdividePreVertexArray = NULL;
        nByte = 1ll * sizeof(VertexNode) * 8 * fixedDepthNodeNum[maxDepth_h];
        CHECK(cudaMalloc((VertexNode**)&SubdividePreVertexArray,nByte));
        CHECK(cudaMemset(SubdividePreVertexArray,0,nByte));

        initSubdivideVertexOwner<<<grid,block>>>(NodeArray_sz,
                                                 SubdivideArray,fixedDepthNodeAddress[maxDepth_h],SubdivideArray_sz,
                                                 SubdividePreVertexArray,
                                                 SubdivideArrayCenterBuffer);
        cudaDeviceSynchronize();


        VertexNode *SubdivideVertexArray = NULL;
//        nByte = 1ll * sizeof(VertexNode) * 8 * fixedDepthNodeNum[maxDepth_h];
        CHECK(cudaMalloc((VertexNode**)&SubdivideVertexArray,nByte));
        CHECK(cudaMemset(SubdivideVertexArray,0,nByte));
        thrust::device_ptr<VertexNode> SubdividePreVertexArray_ptr = thrust::device_pointer_cast<VertexNode>(SubdividePreVertexArray);
        thrust::device_ptr<VertexNode> SubdivideVertexArray_ptr = thrust::device_pointer_cast<VertexNode>(SubdivideVertexArray);
        thrust::device_ptr<VertexNode> SubdivideVertexArray_end = thrust::copy_if(SubdividePreVertexArray_ptr, SubdividePreVertexArray_ptr + 8 * fixedDepthNodeNum[maxDepth_h],SubdivideVertexArray_ptr,validVertex());
        cudaDeviceSynchronize();

        cudaFree(SubdividePreVertexArray);

        int SubdivideVertexArray_sz = SubdivideVertexArray_end - SubdivideVertexArray_ptr;

//        for(int j=0;j<10;++j){
//            printf("%d owner:%d\n",j,SubdivideVertexArray[j].ownerNodeIdx);
//        }

        maintainSubdivideVertexNodePointer<<<grid,block>>>(SubdivideVertexArray,SubdivideVertexArray_sz,
                                                           NodeArray_sz,
                                                           SubdivideArray,
                                                           CenterBuffer,
                                                           SubdivideArrayCenterBuffer);
        cudaDeviceSynchronize();



        // ----------------------------------------------------

        // preEdgeArray
        EdgeNode *SubdividePreEdgeArray = NULL;
        nByte = 1ll * sizeof(EdgeNode) * 12 * fixedDepthNodeNum[maxDepth_h];
        CHECK(cudaMalloc((EdgeNode**)&SubdividePreEdgeArray,nByte));
        CHECK(cudaMemset(SubdividePreEdgeArray,0,nByte));

        initSubdivideEdgeArray<<<grid,block>>>(SubdivideArray,fixedDepthNodeAddress[maxDepth_h],SubdivideArray_sz,
                                               NodeArray_sz,
                                               SubdividePreEdgeArray,
                                               SubdivideArrayCenterBuffer);
        cudaDeviceSynchronize();

        EdgeNode *SubdivideEdgeArray=NULL;
        CHECK(cudaMalloc((EdgeNode**)&SubdivideEdgeArray,nByte));
        CHECK(cudaMemset(SubdivideEdgeArray,0,nByte));

        thrust::device_ptr<EdgeNode> SubdividePreEdgeArray_ptr = thrust::device_pointer_cast<EdgeNode>(SubdividePreEdgeArray);
        thrust::device_ptr<EdgeNode> SubdivideEdgeArray_ptr = thrust::device_pointer_cast<EdgeNode>(SubdivideEdgeArray);
        thrust::device_ptr<EdgeNode> SubdivideEdgeArray_end = thrust::copy_if(SubdividePreEdgeArray_ptr,SubdividePreEdgeArray_ptr + 12 * fixedDepthNodeNum[maxDepth_h],SubdivideEdgeArray_ptr,validEdge());
        cudaDeviceSynchronize();

        cudaFree(SubdividePreEdgeArray);

        int SubdivideEdgeArray_sz = SubdivideEdgeArray_end - SubdivideEdgeArray_ptr;

        maintainSubdivideEdgeNodePointer<<<grid,block>>>(SubdivideEdgeArray,SubdivideEdgeArray_sz,
                                                         NodeArray_sz,
                                                         SubdivideArray,
                                                         CenterBuffer,
                                                         SubdivideArrayCenterBuffer);
        cudaDeviceSynchronize();

//        for(int j=0;j<15;++j){
//            printf("%d owner:%d\n",j,SubdivideEdgeArray[j].ownerNodeIdx);
//        }
//
//        for (int j = fixedDepthNodeAddress[maxDepth_h]; j < fixedDepthNodeAddress[maxDepth_h] + 10; ++j) {
//            std::cout << std::bitset<32>(SubdivideArray[j].key) << " parent:" << SubdivideArray[j].parent
//                      << std::endl;
//            for (int k = 0; k < 8; ++k) {
//                std::cout << "children[" << k << "]:" << SubdivideArray[j].children[k] << " ";
//            }
//            std::cout << std::endl;
//            for (int k = 0; k < 27; ++k) {
//                std::cout << "neigh:[" << k << "]:" << SubdivideArray[j].neighs[k] << " ";
//            }
//            std::cout << std::endl;
//            for(int k=0;k<8;++k){
//                std::cout << "vertices["<<k<<"]:"<<SubdivideArray[j].vertices[k]<<" ";
//            }
//            std::cout << std::endl;
//            for(int k=0;k<12;++k){
//                std::cout << "edges["<<k<<"]:"<<SubdivideArray[j].edges[k]<<" ";
//            }
//            std::cout << std::endl;
//        }

        // ----------------------------------------------------

        float *SubdivideVvalue = NULL;
        nByte = 1ll * sizeof(float) * SubdivideVertexArray_sz;
        CHECK(cudaMalloc((float**)&SubdivideVvalue,nByte));
        CHECK(cudaMemset(SubdivideVvalue,0,nByte));

        computeSubdivideVertexImplicitFunctionValue<<<grid,block>>>(SubdivideVertexArray,SubdivideVertexArray_sz,
                                                                    SubdivideArray,rootId,
                                                                    NodeArray,NodeArray_sz,
                                                                    d_x,EncodedNodeIdxInFunction,
                                                                    baseFunctions_d,
                                                                    SubdivideVvalue,isoValue);
        cudaDeviceSynchronize();

        int *SubdivideVexNums=NULL;
        nByte = 1ll * sizeof(int) * SubdivideEdgeArray_sz;
        CHECK(cudaMalloc((int**)&SubdivideVexNums,nByte));
        CHECK(cudaMemset(SubdivideVexNums,0,nByte));

        generateSubdivideVexNums<<<grid,block>>>(SubdivideEdgeArray,SubdivideEdgeArray_sz,
                                                 NodeArray_sz,
                                                 SubdivideArray,SubdivideVvalue,
                                                 SubdivideVexNums);
        cudaDeviceSynchronize();

        int *SubdivideVexAddress=NULL;
        nByte = 1ll * sizeof(int) *SubdivideEdgeArray_sz;
        CHECK(cudaMalloc((int**)&SubdivideVexAddress,nByte));
        CHECK(cudaMemset(SubdivideVexAddress,0,nByte));

        thrust::device_ptr<int> SubdivideVexNums_ptr = thrust::device_pointer_cast<int>(SubdivideVexNums);
        thrust::device_ptr<int> SubdivideVexAddress_ptr = thrust::device_pointer_cast<int>(SubdivideVexAddress);

        thrust::exclusive_scan(SubdivideVexNums_ptr,SubdivideVexNums_ptr + SubdivideEdgeArray_sz, SubdivideVexAddress_ptr);
        cudaDeviceSynchronize();

//        int SubdivideLastVexAddr = SubdivideVexAddress[SubdivideEdgeArray_sz-1];
//        int SubdivideLastVexNums = SubdivideVexNums[SubdivideEdgeArray_sz-1];
        int SubdivideLastVexAddr;
        int SubdivideLastVexNums;
        CHECK(cudaMemcpy(&SubdivideLastVexAddr,SubdivideVexAddress+SubdivideEdgeArray_sz-1,sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&SubdivideLastVexNums,SubdivideVexNums+SubdivideEdgeArray_sz-1,sizeof(int),cudaMemcpyDeviceToHost));

        int SubdivideAllVexNums = SubdivideLastVexAddr + SubdivideLastVexNums;

        if(SubdivideAllVexNums == 0){
            CHECK(cudaMemcpy(&NodeArray[rootParent].children[rootSonKey],&rootId,sizeof(int),cudaMemcpyHostToDevice));
            cudaFree(SubdivideVertexArray);
            cudaFree(SubdivideEdgeArray);
            cudaFree(SubdivideVvalue);
            cudaFree(SubdivideVexNums);
            cudaFree(SubdivideVexAddress);
            continue;
        }

        int *SubdivideTriNums=NULL;
        nByte = 1ll * sizeof(int) * fixedDepthNodeNum[maxDepth_h];
        CHECK(cudaMalloc((int**)&SubdivideTriNums,nByte));
        CHECK(cudaMemset(SubdivideTriNums,0,nByte));

        int *SubdivideCubeCatagory = NULL;
        nByte = 1ll * sizeof(int) * fixedDepthNodeNum[maxDepth_h];
        CHECK(cudaMalloc((int**)&SubdivideCubeCatagory,nByte));
        CHECK(cudaMemset(SubdivideCubeCatagory,0,nByte));

        generateTriNums<<<grid,block>>>(SubdivideArray,
                                        fixedDepthNodeAddress[maxDepth_h],SubdivideArray_sz,
                                        SubdivideVvalue,
                                        SubdivideTriNums,SubdivideCubeCatagory);
        cudaDeviceSynchronize();

        int *SubdivideTriAddress=NULL;
        nByte = 1ll * sizeof(int) * fixedDepthNodeNum[maxDepth_h];
        CHECK(cudaMalloc((int**)&SubdivideTriAddress,nByte));
        CHECK(cudaMemset(SubdivideTriAddress,0,nByte));

        thrust::device_ptr<int> SubdivideTriNums_ptr = thrust::device_pointer_cast<int>(SubdivideTriNums);
        thrust::device_ptr<int> SubdivideTriAddress_ptr = thrust::device_pointer_cast<int>(SubdivideTriAddress);

        thrust::exclusive_scan(SubdivideTriNums_ptr,SubdivideTriNums_ptr + fixedDepthNodeNum[maxDepth_h],SubdivideTriAddress_ptr);
        cudaDeviceSynchronize();

        Point3D<float> *SubdivideVertexBuffer = NULL;
        nByte = 1ll * sizeof(Point3D<float>) * SubdivideAllVexNums;
        CHECK(cudaMallocManaged((Point3D<float>**)&SubdivideVertexBuffer,nByte));

        EdgeNode * SubdivideValidEdgeArray = NULL;
        nByte = 1ll * sizeof(EdgeNode) * SubdivideAllVexNums;
        CHECK(cudaMalloc((EdgeNode**)&SubdivideValidEdgeArray,nByte));
        thrust::device_ptr<EdgeNode> SubdivideValidEdgeArray_ptr = thrust::device_pointer_cast<EdgeNode>(SubdivideValidEdgeArray);
        SubdivideEdgeArray_ptr = thrust::device_pointer_cast<EdgeNode>(SubdivideEdgeArray);
        thrust::device_ptr<EdgeNode> SubdivideValidEdgeArray_end = thrust::copy_if(SubdivideEdgeArray_ptr,SubdivideEdgeArray_ptr + SubdivideEdgeArray_sz,SubdivideVexNums_ptr,SubdivideValidEdgeArray_ptr,validVexNums());
        cudaDeviceSynchronize();

        int *SubdivideValidVexAddress = NULL;
        nByte = 1ll * sizeof(int) * SubdivideAllVexNums;
        CHECK(cudaMalloc((int**)&SubdivideValidVexAddress,nByte));
        thrust::device_ptr<int> SubdivideValidVexAddress_ptr = thrust::device_pointer_cast<int>(SubdivideValidVexAddress);
        thrust::device_ptr<int> SubdivideValidVexAddress_end = thrust::copy_if(SubdivideVexAddress_ptr,SubdivideVexAddress_ptr + SubdivideEdgeArray_sz,SubdivideVexNums_ptr,SubdivideValidVexAddress_ptr,validVexNums());
        cudaDeviceSynchronize();

//        grid = (std::min(SubdivideAllVexNums / 1024 , 32) ,SubdivideAllVexNums/32768);
        generateSubdivideIntersectionPoint<<<grid,block>>>(SubdivideValidEdgeArray,SubdivideAllVexNums,
                                                           SubdivideVertexArray,SubdivideArray,
                                                           NodeArray_sz,
                                                           SubdivideValidVexAddress,SubdivideVvalue,
                                                           SubdivideVertexBuffer);
        cudaDeviceSynchronize();

        cudaFree(SubdivideValidEdgeArray);
        cudaFree(SubdivideValidVexAddress);


//        int SubdivideLastTriAddr = SubdivideTriAddress[fixedDepthNodeNum[maxDepth_h]-1];
//        int SubdivideLastTriNums = SubdivideTriNums[fixedDepthNodeNum[maxDepth_h]-1];
        int SubdivideLastTriAddr;
        int SubdivideLastTriNums;
        CHECK(cudaMemcpy(&SubdivideLastTriAddr,SubdivideTriAddress+fixedDepthNodeNum[maxDepth_h]-1,sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&SubdivideLastTriNums,SubdivideTriNums+fixedDepthNodeNum[maxDepth_h]-1,sizeof(int),cudaMemcpyDeviceToHost));
        int SubdivideAllTriNums = SubdivideLastTriAddr + SubdivideLastTriNums;


        int *SubdivideTriangleBuffer=NULL;
        nByte = 1ll * sizeof(int) * 3 * SubdivideAllTriNums;
        CHECK(cudaMallocManaged((int**)&SubdivideTriangleBuffer,nByte));

        generateSubdivideTrianglePos<<<grid,block>>>(SubdivideArray,fixedDepthNodeAddress[maxDepth_h],SubdivideArray_sz,
                                                     SubdivideTriNums,SubdivideCubeCatagory,
                                                     SubdivideVexAddress,
                                                     SubdivideTriAddress,SubdivideTriangleBuffer);
        cudaDeviceSynchronize();


        insertTriangle(SubdivideVertexBuffer,SubdivideAllVexNums,
                       SubdivideTriangleBuffer,SubdivideAllTriNums,
                       mesh);

//        printf("SubdivideAllVexNums:%d SubdivideAllTriNums:%d\n",SubdivideAllVexNums,SubdivideAllTriNums);



        CHECK(cudaMemcpy(&NodeArray[rootParent].children[rootSonKey],&rootId,sizeof(int),cudaMemcpyHostToDevice));
        cudaFree(SubdivideVertexArray);
        cudaFree(SubdivideEdgeArray);
        cudaFree(SubdivideVvalue);
        cudaFree(SubdivideVexNums);
        cudaFree(SubdivideVexAddress);
        cudaFree(SubdivideTriNums);
        cudaFree(SubdivideCubeCatagory);
        cudaFree(SubdivideTriAddress);
        cudaFree(SubdivideVertexBuffer);
        cudaFree(SubdivideTriangleBuffer);
    }

    cudaFree(SubdivideArray);
    cudaFree(SubdivideArrayCenterBuffer);
    cudaFree(SubdivideArrayDepthBuffer);

    double mid14 = cpuSecond();
    printf("GPU processing coarse subdivide nodes takes %lfs\n",mid14-mid13);

    for(int i = finerDepth;i<maxDepth_h;++i) {
        printf("Start the %d depth finer subdivide nodes processing\n",i);
        int *fixedDepthNums = NULL;
        int finerDepthStart = SubdivideDepthAddress[i];
        int finerSubdivideNum = SubdivideDepthCount[i];
        nByte = 1ll * sizeof(int) * finerSubdivideNum * maxDepth_h;
        CHECK(cudaMalloc((int **) &fixedDepthNums, nByte));
        CHECK(cudaMemset(fixedDepthNums, 0, nByte));

        initFixedDepthNums<<<grid, block>>>(SubdivideNode, finerDepthStart, finerDepthStart + finerSubdivideNum,
                                            SubdivideDepthBuffer,
                                            fixedDepthNums);
        cudaDeviceSynchronize();

        thrust::device_ptr<int> fixedDepthNums_ptr = thrust::device_pointer_cast<int>(fixedDepthNums);
        int rebuildNums = thrust::reduce(fixedDepthNums_ptr, fixedDepthNums_ptr + finerSubdivideNum * maxDepth_h);
        cudaDeviceSynchronize();

        int depthNodeCount[maxDepth_h + 1];
        depthNodeCount[0] = 0;
        for (int depth = 1; depth <= maxDepth_h; ++depth) {
            depthNodeCount[depth] = thrust::reduce(fixedDepthNums_ptr + (depth - 1) * finerSubdivideNum,
                                                   fixedDepthNums_ptr + depth * finerSubdivideNum);
        }
        cudaDeviceSynchronize();

        int depthNodeAddress[maxDepth_h + 1];
        depthNodeAddress[0] = 0;
        for (int depth = 1; depth <= maxDepth_h; ++depth) {
            depthNodeAddress[depth] = depthNodeAddress[depth - 1] + depthNodeCount[depth - 1];
//            printf("%d %d %d\n", depth, depthNodeAddress[depth], depthNodeCount[depth]);
        }
//        printf("rebuildNums:%d\n", rebuildNums);

        int *depthNodeAddress_d = NULL;
        nByte = 1ll * sizeof(int) * (maxDepth_h + 1);
        CHECK(cudaMalloc((int **) &depthNodeAddress_d, nByte));
        CHECK(cudaMemcpy(depthNodeAddress_d, depthNodeAddress, nByte, cudaMemcpyHostToDevice));

        int *fixedDepthAddress = NULL;
        nByte = 1ll * sizeof(int) * finerSubdivideNum * maxDepth_h;
        CHECK(cudaMalloc((int **) &fixedDepthAddress, nByte));
        CHECK(cudaMemset(fixedDepthAddress, 0, nByte));
        for (int depth = 1; depth <= maxDepth_h; ++depth) {
            thrust::device_ptr<int> fixedDepthAddress_ptr = thrust::device_pointer_cast<int>(
                    fixedDepthAddress + (depth - 1) * finerSubdivideNum);
            thrust::exclusive_scan(fixedDepthNums_ptr + (depth - 1) * finerSubdivideNum,
                                   fixedDepthNums_ptr + depth * finerSubdivideNum, fixedDepthAddress_ptr);
            cudaDeviceSynchronize();
        }

        EasyOctNode *RebuildArray = NULL;
        long long nBytell = 1ll * sizeof(EasyOctNode) * rebuildNums;
        CHECK(cudaMalloc((EasyOctNode **) &RebuildArray, nBytell));
        CHECK(cudaMemset(RebuildArray, 0, nBytell));

        int *RebuildDepthBuffer = NULL;
        nBytell = 1ll * sizeof(int) * rebuildNums;
        CHECK(cudaMalloc((int **) &RebuildDepthBuffer, nBytell));

        Point3D<float> *RebuildCenterBuffer = NULL;
        nBytell = 1ll * sizeof(Point3D<float>) * rebuildNums;
        CHECK(cudaMalloc((Point3D<float> **) &RebuildCenterBuffer, nBytell));

        int *ReplaceNodeId = NULL;
        nBytell = 1ll * sizeof(int) * rebuildNums;
        CHECK(cudaMalloc((int **) &ReplaceNodeId, nBytell));

        int *IsRoot = NULL;
        nBytell = 1ll * sizeof(int) * rebuildNums;
        CHECK(cudaMalloc((int **) &IsRoot, nBytell));
        CHECK(cudaMemset(IsRoot, 0, nBytell));


        wholeRebuildArray<<<grid, block>>>(SubdivideNode, finerDepthStart, finerDepthStart + finerSubdivideNum,
                                           NodeArray, NodeArray_sz,
                                           SubdivideDepthBuffer,
                                           depthNodeAddress_d, fixedDepthAddress,
                                           RebuildArray,
                                           RebuildDepthBuffer, RebuildCenterBuffer,
                                           ReplaceNodeId, IsRoot);
        cudaDeviceSynchronize();


        for (int j = finerDepth; j <= maxDepth_h; ++j) {
            computeRebuildNeighbor<<<grid, block>>>(RebuildArray, depthNodeAddress[j],
                                                    depthNodeAddress[j] + depthNodeCount[j],
                                                    NodeArray, NodeArray_sz,
                                                    j);
            cudaDeviceSynchronize();

        }

        // preVertexArray
        VertexNode *RebuildPreVertexArray = NULL;
        nByte = 1ll * sizeof(VertexNode) * 8 * depthNodeCount[maxDepth_h];
        CHECK(cudaMalloc((VertexNode **) &RebuildPreVertexArray, nByte));
        CHECK(cudaMemset(RebuildPreVertexArray, 0, nByte));

        initSubdivideVertexOwner<<<grid, block>>>(NodeArray_sz,
                                                  RebuildArray, depthNodeAddress[maxDepth_h], rebuildNums,
                                                  RebuildPreVertexArray,
                                                  RebuildCenterBuffer);
        cudaDeviceSynchronize();

        VertexNode *RebuildVertexArray = NULL;
        CHECK(cudaMalloc((VertexNode **) &RebuildVertexArray, nByte));
        CHECK(cudaMemset(RebuildVertexArray, 0, nByte));
        thrust::device_ptr<VertexNode> RebuildPreVertexArray_ptr = thrust::device_pointer_cast<VertexNode>(
                RebuildPreVertexArray);
        thrust::device_ptr<VertexNode> RebuildVertexArray_ptr = thrust::device_pointer_cast<VertexNode>(
                RebuildVertexArray);
        thrust::device_ptr<VertexNode> RebuildVertexArray_end = thrust::copy_if(RebuildPreVertexArray_ptr,
                                                                                RebuildPreVertexArray_ptr +
                                                                                8 * depthNodeCount[maxDepth_h],
                                                                                RebuildVertexArray_ptr, validVertex());
        cudaDeviceSynchronize();

        cudaFree(RebuildPreVertexArray);

        int RebuildVertexArray_sz = RebuildVertexArray_end - RebuildVertexArray_ptr;

//        printf("RebuildVertexArray_sz:%d\n", RebuildVertexArray_sz);


        maintainSubdivideVertexNodePointer<<<grid, block>>>(RebuildVertexArray, RebuildVertexArray_sz,
                                                            NodeArray_sz,
                                                            RebuildArray,
                                                            CenterBuffer,
                                                            RebuildCenterBuffer);
        cudaDeviceSynchronize();

//    for (int t = depthNodeAddress[maxDepth_h]; t < depthNodeAddress[maxDepth_h]+10; ++t) {
//        std::cout << t<<std::endl;
//        std::cout << std::bitset<32>(RebuildArray[t].key) << " parent:" << RebuildArray[t].parent
//                  << std::endl;
//        for (int k = 0; k < 8; ++k) {
//            std::cout << "children[" << k << "]:" << RebuildArray[t].children[k] << " ";
//        }
//        std::cout << std::endl;
//        for (int k = 0; k < 27; ++k) {
//            std::cout << "neigh:[" << k << "]" << RebuildArray[t].neighs[k] << " ";
//        }
//        std::cout << std::endl;
//        for (int k = 0; k < 8; ++k) {
//            std::cout << "vertices:[" << k << "]" << RebuildArray[t].vertices[k] << " ";
//        }
//        std::cout << std::endl;
//    }


        // preEdgeArray
        EdgeNode *RebuildPreEdgeArray = NULL;
        nByte = 1ll * sizeof(EdgeNode) * 12 * depthNodeCount[maxDepth_h];
        CHECK(cudaMalloc((EdgeNode **) &RebuildPreEdgeArray, nByte));
        CHECK(cudaMemset(RebuildPreEdgeArray, 0, nByte));

        initSubdivideEdgeArray<<<grid, block>>>(RebuildArray, depthNodeAddress[maxDepth_h], rebuildNums,
                                                NodeArray_sz,
                                                RebuildPreEdgeArray,
                                                RebuildCenterBuffer);
        cudaDeviceSynchronize();

        EdgeNode *RebuildEdgeArray = NULL;
        CHECK(cudaMalloc((EdgeNode **) &RebuildEdgeArray, nByte));
        CHECK(cudaMemset(RebuildEdgeArray, 0, nByte));

        thrust::device_ptr<EdgeNode> RebuildPreEdgeArray_ptr = thrust::device_pointer_cast<EdgeNode>(
                RebuildPreEdgeArray);
        thrust::device_ptr<EdgeNode> RebuildEdgeArray_ptr = thrust::device_pointer_cast<EdgeNode>(RebuildEdgeArray);
        thrust::device_ptr<EdgeNode> RebuildEdgeArray_end = thrust::copy_if(RebuildPreEdgeArray_ptr,
                                                                            RebuildPreEdgeArray_ptr +
                                                                            12 * depthNodeCount[maxDepth_h],
                                                                            RebuildEdgeArray_ptr, validEdge());
        cudaDeviceSynchronize();

        cudaFree(RebuildPreEdgeArray);

        int RebuildEdgeArray_sz = RebuildEdgeArray_end - RebuildEdgeArray_ptr;


        maintainSubdivideEdgeNodePointer<<<grid, block>>>(RebuildEdgeArray, RebuildEdgeArray_sz,
                                                          NodeArray_sz,
                                                          RebuildArray,
                                                          CenterBuffer,
                                                          RebuildCenterBuffer);
        cudaDeviceSynchronize();


        float *RebuildVvalue = NULL;
        nByte = 1ll * sizeof(float) * RebuildVertexArray_sz;
        CHECK(cudaMalloc((float **) &RebuildVvalue, nByte));
        CHECK(cudaMemset(RebuildVvalue, 0, nByte));

        computeSubdivideVertexImplicitFunctionValue<<<grid, block>>>(RebuildVertexArray, RebuildVertexArray_sz,
                                                                     RebuildArray, ReplaceNodeId, IsRoot,
                                                                     NodeArray, NodeArray_sz,
                                                                     d_x, EncodedNodeIdxInFunction,
                                                                     baseFunctions_d,
                                                                     RebuildVvalue, isoValue);
        cudaDeviceSynchronize();

        cudaFree(ReplaceNodeId);
        cudaFree(IsRoot);

        int *RebuildVexNums = NULL;
        nByte = 1ll * sizeof(int) * RebuildEdgeArray_sz;
        CHECK(cudaMalloc((int **) &RebuildVexNums, nByte));
        CHECK(cudaMemset(RebuildVexNums, 0, nByte));

        generateSubdivideVexNums<<<grid, block>>>(RebuildEdgeArray, RebuildEdgeArray_sz,
                                                  NodeArray_sz,
                                                  RebuildArray, RebuildVvalue,
                                                  RebuildVexNums);
        cudaDeviceSynchronize();


        int *RebuildVexAddress = NULL;
        nByte = 1ll * sizeof(int) * RebuildEdgeArray_sz;
        CHECK(cudaMalloc((int **) &RebuildVexAddress, nByte));
        CHECK(cudaMemset(RebuildVexAddress, 0, nByte));

        thrust::device_ptr<int> RebuildVexNums_ptr = thrust::device_pointer_cast<int>(RebuildVexNums);
        thrust::device_ptr<int> RebuildVexAddress_ptr = thrust::device_pointer_cast<int>(RebuildVexAddress);

        thrust::exclusive_scan(RebuildVexNums_ptr, RebuildVexNums_ptr + RebuildEdgeArray_sz, RebuildVexAddress_ptr);
        cudaDeviceSynchronize();

        int RebuildLastVexAddr;
        int RebuildLastVexNums;
        CHECK(cudaMemcpy(&RebuildLastVexAddr, RebuildVexAddress + RebuildEdgeArray_sz - 1, sizeof(int),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&RebuildLastVexNums, RebuildVexNums + RebuildEdgeArray_sz - 1, sizeof(int),
                         cudaMemcpyDeviceToHost));

        int RebuildAllVexNums = RebuildLastVexAddr + RebuildLastVexNums;

//        printf("RebuildAllVexNums:%d\n", RebuildAllVexNums);


        int *RebuildTriNums = NULL;
        nByte = 1ll * sizeof(int) * depthNodeCount[maxDepth_h];
        CHECK(cudaMalloc((int **) &RebuildTriNums, nByte));
        CHECK(cudaMemset(RebuildTriNums, 0, nByte));

        int *RebuildCubeCatagory = NULL;
        nByte = 1ll * sizeof(int) * depthNodeCount[maxDepth_h];
        CHECK(cudaMalloc((int **) &RebuildCubeCatagory, nByte));
        CHECK(cudaMemset(RebuildCubeCatagory, 0, nByte));

        generateTriNums<<<grid, block>>>(RebuildArray,
                                         depthNodeAddress[maxDepth_h], rebuildNums,
                                         RebuildVvalue,
                                         RebuildTriNums, RebuildCubeCatagory);
        cudaDeviceSynchronize();

        int *RebuildTriAddress = NULL;
        nByte = 1ll * sizeof(int) * depthNodeCount[maxDepth_h];
        CHECK(cudaMalloc((int **) &RebuildTriAddress, nByte));
        CHECK(cudaMemset(RebuildTriAddress, 0, nByte));

        thrust::device_ptr<int> RebuildTriNums_ptr = thrust::device_pointer_cast<int>(RebuildTriNums);
        thrust::device_ptr<int> RebuildTriAddress_ptr = thrust::device_pointer_cast<int>(RebuildTriAddress);

        thrust::exclusive_scan(RebuildTriNums_ptr, RebuildTriNums_ptr + depthNodeCount[maxDepth_h],
                               RebuildTriAddress_ptr);
        cudaDeviceSynchronize();

        Point3D<float> *RebuildVertexBuffer = NULL;
        nByte = 1ll * sizeof(Point3D<float>) * RebuildAllVexNums;
        CHECK(cudaMallocManaged((Point3D<float> **) &RebuildVertexBuffer, nByte));


        EdgeNode *RebuildValidEdgeArray = NULL;
        nByte = 1ll * sizeof(EdgeNode) * RebuildAllVexNums;
        CHECK(cudaMalloc((EdgeNode **) &RebuildValidEdgeArray, nByte));
        thrust::device_ptr<EdgeNode> RebuildValidEdgeArray_ptr = thrust::device_pointer_cast<EdgeNode>(
                RebuildValidEdgeArray);
        RebuildEdgeArray_ptr = thrust::device_pointer_cast<EdgeNode>(RebuildEdgeArray);
        thrust::device_ptr<EdgeNode> RebuildValidEdgeArray_end = thrust::copy_if(RebuildEdgeArray_ptr,
                                                                                 RebuildEdgeArray_ptr +
                                                                                 RebuildEdgeArray_sz,
                                                                                 RebuildVexNums_ptr,
                                                                                 RebuildValidEdgeArray_ptr,
                                                                                 validVexNums());
        cudaDeviceSynchronize();

        int *RebuildValidVexAddress = NULL;
        nByte = 1ll * sizeof(int) * RebuildAllVexNums;
        CHECK(cudaMalloc((int **) &RebuildValidVexAddress, nByte));
        thrust::device_ptr<int> RebuildValidVexAddress_ptr = thrust::device_pointer_cast<int>(RebuildValidVexAddress);
        thrust::device_ptr<int> RebuildValidVexAddress_end = thrust::copy_if(RebuildVexAddress_ptr,
                                                                             RebuildVexAddress_ptr +
                                                                             RebuildEdgeArray_sz, RebuildVexNums_ptr,
                                                                             RebuildValidVexAddress_ptr,
                                                                             validVexNums());
        cudaDeviceSynchronize();

        generateSubdivideIntersectionPoint<<<grid, block>>>(RebuildValidEdgeArray, RebuildAllVexNums,
                                                            RebuildVertexArray, RebuildArray,
                                                            NodeArray_sz,
                                                            RebuildValidVexAddress, RebuildVvalue,
                                                            RebuildVertexBuffer);
        cudaDeviceSynchronize();

        cudaFree(RebuildValidEdgeArray);
        cudaFree(RebuildValidVexAddress);
        cudaFree(RebuildVvalue);


        int RebuildLastTriAddr;
        int RebuildLastTriNums;
//        RebuildLastTriAddr = RebuildTriAddress[depthNodeCount[maxDepth_h] - 1];
//        RebuildLastTriNums = RebuildTriNums[depthNodeCount[maxDepth_h] - 1];
        CHECK(cudaMemcpy(&RebuildLastTriAddr,RebuildTriAddress+depthNodeCount[maxDepth_h]-1,sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&RebuildLastTriNums,RebuildTriNums+depthNodeCount[maxDepth_h]-1,sizeof(int),cudaMemcpyDeviceToHost));
        int RebuildAllTriNums = RebuildLastTriAddr + RebuildLastTriNums;

//        printf("RebuildAllTriNums:%d\n", RebuildAllTriNums);

        cudaFree(RebuildVexNums);

        int *RebuildTriangleBuffer = NULL;
        nByte = 1ll * sizeof(int) * 3 * RebuildAllTriNums;
        CHECK(cudaMallocManaged((int **) &RebuildTriangleBuffer, nByte));

        generateSubdivideTrianglePos<<<grid, block>>>(RebuildArray, depthNodeAddress[maxDepth_h], rebuildNums,
                                                      RebuildTriNums, RebuildCubeCatagory,
                                                      RebuildVexAddress,
                                                      RebuildTriAddress, RebuildTriangleBuffer);
        cudaDeviceSynchronize();

        insertTriangle(RebuildVertexBuffer, RebuildAllVexNums,
                       RebuildTriangleBuffer, RebuildAllTriNums,
                       mesh);

        cudaFree(fixedDepthNums);
        cudaFree(depthNodeAddress_d);
        cudaFree(fixedDepthAddress);
        cudaFree(RebuildArray);
        cudaFree(RebuildDepthBuffer);
        cudaFree(RebuildCenterBuffer);
        cudaFree(RebuildVertexArray);
        cudaFree(RebuildEdgeArray);
        cudaFree(RebuildVexAddress);
        cudaFree(RebuildTriNums);
        cudaFree(RebuildCubeCatagory);
        cudaFree(RebuildTriAddress);
        cudaFree(RebuildVertexBuffer);
        cudaFree(RebuildTriangleBuffer);
    }

    double mid15 = cpuSecond();
    printf("GPU processing finer subdivide nodes takes %lfs\n",mid15-mid14);

    PlyWriteTriangles(outName,&mesh, PLY_ASCII,center,scale,NULL,0);

    double mid16 = cpuSecond();
    printf("Output ply files takes %lfs\n",mid16-mid15);

    printf("The whole project takes %lfs (including I/O)\n",mid16-project_st);

}