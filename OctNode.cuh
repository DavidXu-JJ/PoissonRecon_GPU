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
};


#endif //GPU_POISSONRECON_OCTNODE_CUH
