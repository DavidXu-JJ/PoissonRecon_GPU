//
// Created by davidxu on 22-7-27.
//

#ifndef GPU_POISSONRECON_HASH_CUH
#define GPU_POISSONRECON_HASH_CUH

struct KeyValue
{
    int key;
    int value;
};

const int kHashTableCapacity = 128 * 1024 * 1024;

const int kEmpty = 0xffffffff;

// 32 bit Murmur3 hash
__device__ int hash(int k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity-1);
}

KeyValue* create_hashtable(){
    KeyValue *hashtable;
    cudaMalloc(&hashtable,sizeof(KeyValue) * kHashTableCapacity);

    static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
    cudaMemset(hashtable,0xff,sizeof(KeyValue) * kHashTableCapacity);

    return hashtable;
}

__device__ void insert(KeyValue *hashtable,const int& key,const int& value){
    int slot = hash(key);
    while(true)
    {
        int prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
        if (prev == kEmpty || prev == key)
        {
            hashtable[slot].value = value;
            return;
        }
        slot = (slot + 1) & (kHashTableCapacity-1);
    }
}

__device__ void insertMin(KeyValue *hashtable,const int& key,const int& value){
    int slot = hash(key);
    while(true)
    {
        int prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
        if(prev == kEmpty)
        {
            atomicExch(&hashtable[slot].value,0x7fffffff);
            atomicMin(&hashtable[slot].value,value);
            return;
        }
        if(prev == key)
        {
            atomicMin(&hashtable[slot].value,value);
            return;
        }
        slot = (slot + 1) & (kHashTableCapacity-1);
    }
}

__device__ void keyAdd(KeyValue *hashtable,const int& key){
    int slot = hash(key);
    while(true)
    {
        int prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
        if(prev == kEmpty)
        {
            atomicAdd(&hashtable[slot].value,1);
        }
        if(prev == kEmpty || prev == key)
        {
            atomicAdd(&hashtable[slot].value,1);
            return;
        }
        slot = (slot + 1) & (kHashTableCapacity-1);
    }
}


__device__ int find(KeyValue *hashtable,const int& key){
    int slot = hash(key);
    while (true)
    {
        if (hashtable[slot].key == key)
        {
            return hashtable[slot].value;
        }
        if (hashtable[slot].key == kEmpty)
        {
            return 0;
        }
        slot = (slot + 1) & (kHashTableCapacity - 1);
    }
}

void destroy_hashtable(KeyValue* pHashTable)
{
    cudaFree(pHashTable);
}



#endif //GPU_POISSONRECON_HASH_CUH
