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
};


#endif //GPU_POISSONRECON_OCTNODE_CUH
