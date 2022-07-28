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

__global__ void generateUniqueNodeArrayD_1(OctNode *uniqueNode,int DSize,KeyValue *keyIndexHash,KeyValue *pidxHash,int depthD,OctNode *uniqueNodeArrayD_1){
    int stride=gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    int offset= (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    for(int i=offset;i<DSize;i+=stride){
        int fatherKey=uniqueNode[i].key & (~ (7<< (3 * (maxDepth-depthD) ) ) );
        int idx=find(keyIndexHash,fatherKey);
        uniqueNodeArrayD_1[idx].key=fatherKey;
        atomicAdd(&uniqueNodeArrayD_1[idx].pnum,uniqueNode[i].pnum);
        uniqueNodeArrayD_1[idx].pidx = find(pidxHash,fatherKey);
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

__host__ void pipelineUniqueNode_D_1(OctNode *uniqueNode_D,int *nodeAddress_D,int uniqueCount_D,int allNodeNums_D,int depthD,
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
    generateUniqueNodeArrayD_1<<<grid,block>>>(uniqueNode_D,uniqueCount_D,keyIndexHash,pidxHash,depthD,uniqueNode_D_1);
    cudaDeviceSynchronize();
    destroy_hashtable(keyIndexHash);
    destroy_hashtable(pidxHash);

    nByte=sizeof(OctNode) *uniqueCount_D_1;
    OctNode *hh=(OctNode*)malloc(nByte);
    cudaMemcpy(hh,uniqueNode_D_1,nByte,cudaMemcpyDeviceToHost);
    for(int i=0;i<uniqueCount_D_1;++i){
        std::cout<<std::bitset<32>(hh[i].key)<<" pidx:"<<hh[i].pidx<<" pnum:"<<hh[i].pnum<<std::endl;
    }

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

//__host__ void pipelineNodeArrayD_1(OctNode *uniqueNode_D,int *nodeAddress_D,int uniqueCount_D,int allNodeNums_D,int depthD,
//                                   OctNode *uniqueNode_D_1,int &uniqueCount_D_1,
//                                   int *NodeAddress_D_1,
//                                   OctNode *NodeArray_D_1,int &allNodeNums_D_1)
//{
//    pipelineUniqueNode_D_1(uniqueNode_D,nodeAddress_D,uniqueCount_D,allNodeNums_D,depthD,
//                           uniqueNode_D_1,uniqueCount_D_1);
//
//    pipelineNodeAddress_D_1(uniqueNode_D_1,uniqueCount_D_1,depthD,
//                            NodeAddress_D_1);
//
//    dim3 grid=(32,32);
//    dim3 block=(32,32);
//    int lastAddrD_1;
//    CHECK(cudaMemcpy(&lastAddrD_1,NodeAddress_D_1+uniqueCount_D_1-1,sizeof(int),cudaMemcpyDeviceToHost));
//    allNodeNums_D_1=lastAddrD_1+8;
//
//    int nByte=sizeof(OctNode) * allNodeNums_D_1;
////    CHECK(cudaMalloc((OctNode **)&NodeArray_D_1, nByte));
////    CHECK(cudaMemset(NodeArray_D_1,0,nByte));
//
//    generateNodeArrayD_1<<<grid,block>>>(uniqueNode_D_1,NodeAddress_D_1,NodeArray_D_1,uniqueCount_D_1,depthD);
//    cudaDeviceSynchronize();
//
//}



struct markCompact{
    __host__ __device__
    bool operator()(const long long x){
        return ( x & (1ll<<markOffset) ) > 0;
    }
};


int main() {
    char fileName[]="/home/davidxu/horse.npts";

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
    thrust::device_ptr<long long> key_ptr=thrust::device_pointer_cast<long long>(key);
    thrust::sort_by_key(key_ptr,key_ptr+count,samplePoints);
    cudaDeviceSynchronize();

    KeyValue* start_hashTable=create_hashtable();
    KeyValue* count_hashTable=create_hashtable();

    generateStartHash<<<grid,block>>>(key, start_hashTable,count);
    generateCountHash<<<grid,block>>>(key, count_hashTable,count);


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
    uniqueCode.resize(uniqueCount_h);

    /**     Create uniqueNode according to uniqueCode   */
    OctNode *uniqueNode=NULL;
    nByte=sizeof(OctNode)*uniqueCount_h;
    CHECK(cudaMalloc((OctNode **)&uniqueNode,nByte));
    long long *uniqueCode_ptr=thrust::raw_pointer_cast(&uniqueCode[0]);
    initUniqueNode<<<grid,block>>>(uniqueCode_ptr,start_hashTable,count_hashTable,uniqueNode,uniqueCount_h);
    cudaDeviceSynchronize();

//    nByte=sizeof(OctNode)*uniqueCount_h;
//    OctNode *hh=(OctNode*) malloc(nByte);
//    cudaMemcpy(hh,uniqueNode,nByte,cudaMemcpyDeviceToHost);
//    for(int i=0;i<uniqueCount_h;++i){
//        std::cout<<std::bitset<32>(hh[i].key)<<" pidx:"<<hh[i].pidx<<" pnum:"<<hh[i].pnum<<std::endl;
//    }

    /**     Step 5: augment uniqueNode      */
    int *nodeNums=NULL;
    int *nodeAddress=NULL;
    nByte=sizeof(int)*uniqueCount_h;
    CHECK(cudaMalloc((int **)&nodeNums,nByte));
    CHECK(cudaMemset(nodeNums,0,nByte));

    CHECK(cudaMalloc((int **)&nodeAddress,nByte));
    CHECK(cudaMemset(nodeAddress,0,nByte));

    generateNodeNums<<<grid,block>>>(uniqueCode_ptr,nodeNums,uniqueCount_h,maxDepth_h);
    cudaDeviceSynchronize();

    thrust::device_ptr<int> nodeNums_ptr=thrust::device_pointer_cast<int>(nodeNums);
    thrust::device_ptr<int> nodeAddress_ptr=thrust::device_pointer_cast<int>(nodeAddress);

    thrust::inclusive_scan(nodeNums_ptr,nodeNums_ptr+uniqueCount_h,nodeAddress_ptr);
    cudaDeviceSynchronize();


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
    OctNode *uniqueNode_D=uniqueNode;
    int *NodeAddress_D=nodeAddress;
    int uniqueCount_D=uniqueCount_h;
    int allNodeNums_D=allNodeNums;
    for(int depthD=maxDepth_h;depthD>=1;--depthD){
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
        pipelineUniqueNode_D_1(uniqueNode_D,NodeAddress_D,uniqueCount_D,allNodeNums_D,depthD,
                               uniqueNode_D_1,uniqueCount_D_1);
        pipelineNodeAddress_D_1(uniqueNode_D_1,uniqueCount_D_1,depthD,
                                NodeAddress_D_1);

        int lastAddrD_1;
        CHECK(cudaMemcpy(&lastAddrD_1,NodeAddress_D_1+uniqueCount_D_1-1,sizeof(int),cudaMemcpyDeviceToHost));
        allNodeNums_D_1=lastAddrD_1+8;

        nByte=sizeof(OctNode)*allNodeNums_D_1;
        CHECK(cudaMalloc((OctNode **)&NodeArray_D_1, nByte));
        CHECK(cudaMemset(NodeArray_D_1,0,nByte));

        generateNodeArrayD_1<<<grid, block>>>(uniqueNode_D_1, NodeAddress_D_1, NodeArray_D_1, uniqueCount_D_1, depthD);
        cudaDeviceSynchronize();

        nByte=sizeof(OctNode) *uniqueCount_D_1;
        OctNode *h=(OctNode*)malloc(nByte);
        cudaMemcpy(h,uniqueNode_D_1,nByte,cudaMemcpyDeviceToHost);
        for(int i=0;i<uniqueCount_D_1;++i){
            std::cout<<std::bitset<32>(h[i].key)<<" pidx:"<<h[i].pidx<<" pnum:"<<h[i].pnum<<std::endl;
        }
        printf("%d %d\n",uniqueCount_D_1, allNodeNums_D_1);
        uniqueNode_D=uniqueNode_D_1;
        NodeAddress_D=NodeAddress_D_1;
        uniqueCount_D=uniqueCount_D_1;
        allNodeNums_D=allNodeNums_D_1;
    }



    double ed=cpuSecond();
    printf("Numbers of points:%d\nNumbers of uniqueCode:%d\n",count,uniqueCount_h);
    printf("GPU:%lfs\n",ed-mid);

    cudaFree(key);
    cudaFree(uniqueNode);
    cudaFree(nodeNums);
    cudaFree(nodeAddress);
    cudaFree(NodeArrayD);
//    cudaFree(uniqueNodeArrayD_1);
//    free(NodeArrayD_1_h);

    destroy_hashtable(start_hashTable);
    destroy_hashtable(count_hashTable);


}
