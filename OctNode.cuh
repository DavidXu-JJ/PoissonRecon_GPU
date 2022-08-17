//
// Created by davidxu on 22-7-26.
//

#ifndef GPU_POISSONRECON_OCTNODE_CUH
#define GPU_POISSONRECON_OCTNODE_CUH

class OctNode{
public:
    int key;
    int pidx;
    int pnum;
    int parent;
    int children[8];
    int neighs[27];
    // record the start in maxDepth NodeArray
    // the first node at maxDepth is index 0
    int didx;
    int dnum;

    // (real idx) + 1,
    // idx start from (0 + 1)
    // encode the vertices idx?
    int vertices[8];

    // (real idx) + 1,
    // idx start from (0 + 1)
    int edges[12];

    // (real idx) + 1
    // idx start from (0 + 1)
    int faces[6];

    int hasTriangle;
    int hasIntersection;
};

class EasyOctNode{
public:
    int key;
    int parent;
    int children[8];
    int neighs[27];

    // (real idx) + 1,
    // idx start from (0 + 1)
    // encode the vertices idx?
    int vertices[8];

    // (real idx) + 1,
    // idx start from (0 + 1)
    int edges[12];

    __device__ EasyOctNode& operator = (const OctNode& n){
        key = n.key;
        parent = n.parent;
#pragma unroll
        for(int i=0;i<8;++i){
            children[i] = n.children[i];
            vertices[i] = n.vertices[i];
        }
#pragma unroll
        for(int i=0;i<27;++i){
            neighs[i] = n.neighs[i];
        }
#pragma unroll
        for(int i=0;i<12;++i){
            edges[i] = n.edges[i];
        }
    }
};

class VertexNode{
public:
    Point3D<float> pos;
    int ownerNodeIdx;
    int vertexKind;
    int depth;
    // encode the nodes?
    int nodes[8];
};

class EdgeNode{
public:
//    int orientation;
//    int off[2];
    int edgeKind;
    int ownerNodeIdx;
    int nodes[4];
};

class FaceNode{
public:
    int faceKind;
    int ownerNodeIdx;
    int hasParentFace;
    int nodes[2];
};

#endif //GPU_POISSONRECON_OCTNODE_CUH
