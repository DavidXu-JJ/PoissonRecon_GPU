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
    int vertices[8];
    // (real idx) + 1,
    // idx start from (0 + 1)
    int edges[12];
};

class VertexNode{
public:
    Point3D<float> pos;
    int ownerNodeIdx;
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

#endif //GPU_POISSONRECON_OCTNODE_CUH
