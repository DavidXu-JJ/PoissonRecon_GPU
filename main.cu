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


//#define FORCE_UNIT_NORMALS 1

// make readable to device  ?
__constant__ float EPSILON=float(1e-6);
__constant__ float ROUND_EPS=float(1e-5);
__constant__ int maxDepth=9;
__constant__ int markOffset=31;

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
        {2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6,6,7,2,0,1,0,2,3,2},
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
        {2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6,6,7,2,0,1,0,2,3,2},
        {1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1},
        {0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0}
};

struct markCompact{
    __host__ __device__
    bool operator()(const long long x){
        return ( x & (1ll<<markOffset) ) > 0;
    }
};

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
        code[0]|=1ll<<markOffset;
        offset+=stride;
    }
    for(int i=offset;i<size;i+=stride){
        if(code[i]>>32 != code[i-1]>>32) {
            code[i] |= 1ll << markOffset;
        }
    }
}

__global__ void initUniqueNode(long long *uniqueCode, KeyValue *keyStart,KeyValue *keyCount,OctNode *uniqueNode, int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        uniqueNode[i].key= int(uniqueCode[i] >> 32 ) ;
        uniqueNode[i].pidx=find(keyStart,uniqueNode[i].key);
        uniqueNode[i].pnum=find(keyCount,uniqueNode[i].key);
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


__global__ void generateNodeArrayD(OctNode *uniqueNode,int *nodeAddress, OctNode *NodeArrayD,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        int idx=nodeAddress[i] + ( uniqueNode[i].key & 7);
        NodeArrayD[idx] = uniqueNode[i];
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
    int nByte=sizeof(OctNode) * uniqueCount_D_1;
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
    int nByte=sizeof(int)*uniqueCount_D_1;
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
            for(int child=0;child<8;++child){
                NodeArray[i].children[child] += BaseAddressArray_d[depth+1];
            }
        }else {
            NodeArray[i].parent += BaseAddressArray_d[depth - 1];
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

            NodeArray[idx].pidx= nowPIdx;
            nowPIdx += NodeArray[idx].pnum;

            NodeArray[idx].didx= nowDIdx;
            nowDIdx += NodeArray[idx].dnum;

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
            if(NodeArray[ parentIdx ].neighs[LUTparent[sonKey][j]] != -1){
                NodeArray[i].neighs[j] = NodeArray[ NodeArray[parentIdx].neighs[LUTparent[sonKey][j]] ].children[LUTchild[sonKey][j]];
            }else{
                NodeArray[i].neighs[j]= -1;
            }
        }
    }
}

__host__ void pipelineBuildNodeArray(char *fileName,int &count,int &NodeArray_sz,
                                     int NodeArrayCount_h[maxDepth_h+1],int BaseAddressArray_h[maxDepth_h+1], //host
                                     Point3D<float> *&samplePoints_d,Point3D<float> *&sampleNormals_d,OctNode *&NodeArray)    //device
{
    count=0;
    PointStream<float>* pointStream;
    char* ext = GetFileExtension(fileName);
    if      (!strcasecmp(ext,"bnpts"))      pointStream = new BinaryPointStream<float>(fileName);
    else if (!strcasecmp(ext,"ply"))        pointStream = new PLYPointStream<float>(fileName);
    else                                    pointStream = new ASCIIPointStream<float>(fileName);

    Point3D<float> position,normal;
    Point3D<float> mx,mn;
    Point3D<float> center;

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
    long long nByte=sizeof(long long)*count;
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
    nByte=sizeof(OctNode)*uniqueCount_h;
    CHECK(cudaMalloc((OctNode **)&uniqueNode,nByte));
    CHECK(cudaMemset(uniqueNode,0,nByte));
    initUniqueNode<<<grid,block>>>(uniqueCode,start_hashTable,count_hashTable,uniqueNode,uniqueCount_h);
    cudaDeviceSynchronize();

    destroy_hashtable(start_hashTable);
    destroy_hashtable(count_hashTable);

    /**     Step 5: augment uniqueNode      */
    int *nodeNums=NULL;
    int *nodeAddress=NULL;
    nByte=sizeof(int)*uniqueCount_h;
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
    nByte=sizeof(OctNode) * allNodeNums;
    CHECK(cudaMalloc((OctNode **)&NodeArrayD, nByte));
    CHECK(cudaMemset(NodeArrayD,0,nByte));
    generateNodeArrayD<<<grid,block>>>(uniqueNode,nodeAddress,NodeArrayD,uniqueCount_h);
    initNodeArrayD_DIdxDnum<<<grid,block>>>(NodeArrayD,allNodeNums);
    cudaDeviceSynchronize();

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
        nByte=sizeof(OctNode) * D_1Nums;
        CHECK(cudaMalloc((OctNode **)&uniqueNode_D_1,nByte));
        CHECK(cudaMemset(uniqueNode_D_1,0,nByte));
        int *NodeAddress_D_1=NULL;
        nByte=sizeof(int) * D_1Nums;
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

            nByte = sizeof(OctNode) * allNodeNums_D_1;
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

//        nByte=sizeof(OctNode) *uniqueCount_D_1;
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
        printf("%d %d\n",BaseAddressArray_h[i],NodeArrayCount_h[i]);
    }

    nByte=sizeof(int)*(maxDepth_h+1);
//    int *NodeArrayCount_d=NULL;
//    CHECK(cudaMalloc((int **)&NodeArrayCount_d,nByte));
//    CHECK(cudaMemcpy(NodeArrayCount_d,NodeArrayCount_h,nByte,cudaMemcpyHostToDevice));
    int *BaseAddressArray_d=NULL;
    CHECK(cudaMalloc((int **)&BaseAddressArray_d,nByte));
    CHECK(cudaMemcpy(BaseAddressArray_d,BaseAddressArray_h,nByte,cudaMemcpyHostToDevice));

    nByte=sizeof(OctNode)*(BaseAddressArray_h[maxDepth_h]+NodeArrayCount_h[maxDepth_h]);
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
//    for(int j=0;j<4;++j) {
//        for (int i = BaseAddressArray_h[j]; i < BaseAddressArray_h[j]+NodeArrayCount_h[j]; ++i) {
////            if(a[i].pnum==0) continue;
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
//        std::cout<<std::endl;
//    }

    double ed=cpuSecond();
    printf("GPU NodeArray build takes:%lfs\n",ed-mid);

}

__host__ int getDepth(const int& idxOfNodeArray,int *BaseAddressArray){
    int depth;
    for(depth=0;depth<maxDepth_h;++depth){
        if(BaseAddressArray[depth] <= idxOfNodeArray && BaseAddressArray[depth+1] > idxOfNodeArray){
            break;
        }
    }
    return depth;
}

__host__ __device__ void getFunctionIdxOfNode(const int& key,int depthD,int idx[3]){
    idx[0]=(1<<depthD)-1;
    idx[1]=idx[0];
    idx[2]=idx[1];
    for(int depth=depthD;depth >= 1;--depth){
        int sonKeyX = ( key >> (3 * (maxDepth_h-depth) + 2) ) & 1;
        int sonKeyY = ( key >> (3 * (maxDepth_h-depth) + 1) ) & 1;
        int sonKeyZ = ( key >> (3 * (maxDepth_h-depth)) ) & 1;
        idx[0] += sonKeyX * (1<<(depthD-depth));
        idx[1] += sonKeyY * (1<<(depthD-depth));
        idx[2] += sonKeyZ * (1<<(depthD-depth));
    }
}

__device__ double F_center_width_Point(const PPolynomial<2> &BaseFunction_d,const Point3D<float> &center,const float &width,const Point3D<float> &point){
//    PPolynomial<2> thisFunction_X = BaseFunction_d.scale(width).shift(center.coords[0]);
//    PPolynomial<2> thisFunction_Y = BaseFunction_d.scale(width).shift(center.coords[1]);
//    PPolynomial<2> thisFunction_Z = BaseFunction_d.scale(width).shift(center.coords[2]);
    PPolynomial<2> thisFunction_X = BaseFunction_d.shift(center.coords[0]);
    PPolynomial<2> thisFunction_Y = BaseFunction_d.shift(center.coords[1]);
    PPolynomial<2> thisFunction_Z = BaseFunction_d.shift(center.coords[2]);
    return thisFunction_X(point.coords[0]) * thisFunction_Y(point.coords[1]) * thisFunction_Z(point.coords[2]);
}

__global__ void computeVectorField(PPolynomial<2> *BaseFunction_d,Point3D<float> *samplePoints_d,Point3D<float> *sampleNormals_d,OctNode *NodeArray,int left,int right,Point3D<float> *VectorField){
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
        const PPolynomial<2> BaseFunction=BaseFunction_d->scale(width);
        for(int j=0;j<27;++j){
            int neigh=NodeArray[i].neighs[j];
            if(neigh!=-1){
                for(int k=0;k<NodeArray[neigh].pnum;++k){
                    int pointIdx=NodeArray[neigh].pidx+k;
                    double weight= F_center_width_Point(BaseFunction,samplePoints_d[pointIdx],width,o_c);
                    int IdxInMaxDepth=i-left;
                    VectorField[IdxInMaxDepth].coords[0] += weight * sampleNormals_d[pointIdx].coords[0];
                    VectorField[IdxInMaxDepth].coords[1] += weight * sampleNormals_d[pointIdx].coords[1];
                    VectorField[IdxInMaxDepth].coords[2] += weight * sampleNormals_d[pointIdx].coords[2];
                }

            }
        }
    }
}

int main() {
//    char fileName[]="/home/davidxu/horse.npts";
    char fileName[]="/home/davidxu/bunny.points.ply";

    int NodeArrayCount_h[maxDepth_h+1];
    int BaseAddressArray[maxDepth_h+1];

    Point3D<float> *samplePoints_d=NULL, *sampleNormals_d=NULL;
    OctNode *NodeArray;
    int count=0;
    int NodeArray_sz=0;

    pipelineBuildNodeArray(fileName,count,NodeArray_sz,
                           NodeArrayCount_h,BaseAddressArray,
                           samplePoints_d,sampleNormals_d,NodeArray );

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


    double cpu_st=cpuSecond();

    PPolynomial<2> ReconstructionFunction = PPolynomial<2>::GaussianApproximation();
    FunctionData<2,double> fData;
    fData.set(maxDepth_h,ReconstructionFunction,normalize,0);
    //  precomputed inner product table may can be optimized to GPU parallel
    fData.setDotTables(fData.DOT_FLAG | fData.D_DOT_FLAG | fData.D2_DOT_FLAG);
    PPolynomial<2> &F=ReconstructionFunction;
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

    int nByte=sizeof(ReconstructionFunction);
    PPolynomial<2> *BaseFunction_d = NULL;
    CHECK(cudaMalloc((PPolynomial<2>**)&BaseFunction_d,nByte));
    CHECK(cudaMemcpy(BaseFunction_d,&ReconstructionFunction,nByte,cudaMemcpyHostToDevice));

    nByte=sizeof(double) * fData.res * fData.res;
    double *dot_F_DF=NULL;
    CHECK(cudaMalloc((double **)&dot_F_DF,nByte));
    CHECK(cudaMemcpy(dot_F_DF,fData.dotTable,nByte,cudaMemcpyHostToDevice));

    double *dot_F_D2F=NULL;
    CHECK(cudaMalloc((double **)&dot_F_D2F,nByte));
    CHECK(cudaMemcpy(dot_F_D2F,fData.d2DotTable,nByte,cudaMemcpyHostToDevice));

    double cpu_ed=cpuSecond();
    printf("CPU generate precomputed inner product table takes:",cpu_ed-cpu_st);

    Point3D<float> *VectorField=NULL;
    nByte=sizeof(Point3D<float>) * NodeArrayCount_h[maxDepth_h];
    CHECK(cudaMalloc((Point3D<float> **)&VectorField,nByte));
    CHECK(cudaMemset(VectorField,0,nByte));

    double st=cpuSecond();
    dim3 grid=(32,32);
    dim3 block(32,32);
    computeVectorField<<<grid,block>>>(BaseFunction_d,samplePoints_d,sampleNormals_d,
                                       NodeArray,BaseAddressArray[maxDepth_h],NodeArray_sz,VectorField);
    cudaDeviceSynchronize();

    double mid=cpuSecond();
    printf("Compute Vector Field takes:%lfs\n",mid-st);

}
