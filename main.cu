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


//#define FORCE_UNIT_NORMALS 1

// make readable to device  ?
__constant__ float EPSILON=float(1e-6);
__constant__ float ROUND_EPS=float(1e-5);
__constant__ int maxDepth=10;
__constant__ int markOffset=31;

const int markOffset_h=31;
const int maxDepth_h=10;

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

__global__ void generateHashTable(int *keyValue,KeyValue *hashTable,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        insert(hashTable,keyValue[i],keyValue[i+size]);
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


__global__ void generateNodeArrayD(OctNode *uniqueNode,int *nodeAddress, OctNode *NodeArray,int size){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        NodeArray[nodeAddress[i] + ( uniqueNode[i].key & 7) ] = uniqueNode[i];
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

__global__ void generateNodeKeyPidxHash(OctNode *uniqueNode,int uniqueCount,int depthD,KeyValue *pidxHash){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<uniqueCount;i+=stride){
        int fatherKey=uniqueNode[i].key & (~ (7<< (3 * (maxDepth-depthD) ) ) );
        insertMin(pidxHash,fatherKey, uniqueNode[i].pidx);
    }
}

__global__ void generateUniqueNodeArrayD_1(OctNode *NodeArray_D,int DSize,KeyValue *keyIndexHash,KeyValue *pidxHash,int depthD,OctNode *uniqueNodeArrayD_1){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<DSize;i+=stride){
        if(NodeArray_D[i].pnum==0 && NodeArray_D[i].key==0){
            continue;
        }
        int fatherKey=NodeArray_D[i].key & (~ (7<< (3 * (maxDepth-depthD) ) ) );
        int sonKey = ( NodeArray_D[i].key >> (3 * (maxDepth-depthD)) ) & 7;
        int idx=find(keyIndexHash,fatherKey);
        uniqueNodeArrayD_1[idx].key=fatherKey;
        atomicAdd(&uniqueNodeArrayD_1[idx].pnum,NodeArray_D[i].pnum);
        uniqueNodeArrayD_1[idx].pidx = find(pidxHash,fatherKey);
        NodeArray_D[i].parent=idx;
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

__global__ void generateNodeArrayD_1(OctNode *uniqueNodeArrayD_1,int *nodeAddressD_1,OctNode *NodeArrayD_1,int size,int depthD){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<size;i+=stride){
        NodeArrayD_1[nodeAddressD_1[i] + ( (uniqueNodeArrayD_1[i].key>>(3*(maxDepth-depthD+1) ) ) & 7) ] = uniqueNodeArrayD_1[i];
    }
}

__host__ void pipelineUniqueNode_D_1(OctNode *uniqueNode_D,int *nodeAddress_D,int uniqueCount_D,OctNode *NodeArray_D,int allNodeNums_D,int depthD,
                                     OctNode *uniqueNode_D_1, int &uniqueCount_D_1)
{
    uniqueCount_D_1=allNodeNums_D/8;
    int nByte=sizeof(OctNode) * uniqueCount_D_1;
//    CHECK(cudaMalloc((OctNode **)&uniqueNode_D_1,nByte));
//    CHECK(cudaMemset(uniqueNode_D_1,0,nByte));
    KeyValue *keyIndexHash=create_hashtable();
    KeyValue *pidxHash=create_hashtable();
    dim3 grid=(32,32);
    dim3 block=(32,32);
    generateNodeKeyIndexHash<<<grid,block>>>(uniqueNode_D,nodeAddress_D,uniqueCount_D,depthD,keyIndexHash);
    cudaDeviceSynchronize();
    generateNodeKeyPidxHash<<<grid,block>>>(uniqueNode_D,uniqueCount_D,depthD,pidxHash);
    cudaDeviceSynchronize();
    generateUniqueNodeArrayD_1<<<grid,block>>>(NodeArray_D,allNodeNums_D,keyIndexHash,pidxHash,depthD,uniqueNode_D_1);
    cudaDeviceSynchronize();
    destroy_hashtable(keyIndexHash);
    destroy_hashtable(pidxHash);
}

void pipelineNodeAddress_D_1(OctNode *uniqueNode_D_1,int uniqueCount_D_1,int depthD,
                             int *NodeAddress_D_1)
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


__host__ void pipelineBuildNodeArray(char *fileName,int &count,
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
        pipelineUniqueNode_D_1(uniqueNode_D,NodeAddress_D,uniqueCount_D,NodeArray_D,allNodeNums_D,depthD,
                               uniqueNode_D_1,uniqueCount_D_1);
        pipelineNodeAddress_D_1(uniqueNode_D_1,uniqueCount_D_1,depthD,
                                NodeAddress_D_1);

        if(depthD>1) {
            int lastAddrD_1;
            CHECK(cudaMemcpy(&lastAddrD_1, NodeAddress_D_1 + uniqueCount_D_1 - 1, sizeof(int), cudaMemcpyDeviceToHost));
            allNodeNums_D_1 = lastAddrD_1 + 8;

            nByte = sizeof(OctNode) * allNodeNums_D_1;
            CHECK(cudaMalloc((OctNode **) &NodeArray_D_1, nByte));
            CHECK(cudaMemset(NodeArray_D_1, 0, nByte));

            generateNodeArrayD_1<<<grid, block>>>(uniqueNode_D_1, NodeAddress_D_1, NodeArray_D_1, uniqueCount_D_1, depthD);
            cudaDeviceSynchronize();
        }else{
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

    int *NodeArrayCount_d=NULL;
    nByte=sizeof(int)*(maxDepth_h+1);
    CHECK(cudaMalloc((int **)&NodeArrayCount_d,nByte));
    CHECK(cudaMemcpy(NodeArrayCount_d,NodeArrayCount_h,nByte,cudaMemcpyHostToDevice));
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

    int NodeArray_sz=(BaseAddressArray_h[maxDepth_h]+NodeArrayCount_h[maxDepth_h]);
    updateParentChildren<<<grid,block>>>(BaseAddressArray_d,NodeArray,NodeArray_sz);
    cudaDeviceSynchronize();

//    OctNode *a=(OctNode *)malloc(sizeof(OctNode)*NodeArray_sz);
//    cudaMemcpy(a,NodeArray,sizeof(OctNode)*(BaseAddressArray_h[maxDepth_h]+NodeArrayCount_h[maxDepth_h]),cudaMemcpyDeviceToHost);
//    for(int j=0;j<4;++j) {
//        for (int i = BaseAddressArray_h[j]; i < BaseAddressArray_h[j+1]; ++i) {
//            std::cout << std::bitset<32>(a[i].key) << " pidx:" << a[i].pidx << " pnum:" << a[i].pnum << " parent:"
//                      << a[i].parent << std::endl;
//            for(int k=0;k<8;++k){
//                printf("children[%d]:%d ",k,a[i].children[k]);
//            }
//            puts("");
//        }
//        std::cout<<std::endl;
//    }

    double ed=cpuSecond();
    printf("GPU NodeArray build takes:%lfs\n",ed-mid);

}



int main() {
//    char fileName[]="/home/davidxu/horse.npts";
    char fileName[]="/home/davidxu/bunny.points.ply";

    int NodeArrayCount_h[maxDepth_h+1];
    int BaseAddressArray[maxDepth_h+1];

    Point3D<float> *samplePoints_d=NULL, *sampleNormals_d=NULL;
    OctNode *NodeArray;
    int count=0;

    pipelineBuildNodeArray(fileName,count,
                           NodeArrayCount_h,BaseAddressArray,
                           samplePoints_d,sampleNormals_d,NodeArray );

//    for(int i=0;i<=maxDepth_h;++i){
//        printf("%d %d\n",NodeArrayCount_h[i],BaseAddressArray[i]);
//    }
//

//    OctNode *a=(OctNode *)malloc(sizeof(OctNode)*(BaseAddressArray[maxDepth_h]+NodeArrayCount_h[maxDepth_h]));
//    cudaMemcpy(a,NodeArray,sizeof(OctNode)*(BaseAddressArray[maxDepth_h]+NodeArrayCount_h[maxDepth_h]),cudaMemcpyDeviceToHost);
//    for(int i=BaseAddressArray[2];i<BaseAddressArray[3];++i){
//        std::cout<<std::bitset<32>(a[i].key)<<" pidx:"<<a[i].pidx<<" pnum:"<<a[i].pnum<<std::endl;
//    }

}
