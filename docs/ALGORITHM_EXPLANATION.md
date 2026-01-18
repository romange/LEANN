# LEANN Algorithm Explanation

## Overview

LEANN (Low-Storage Vector Index) is an innovative vector database system that achieves **97% storage reduction** compared to traditional vector databases without sacrificing search accuracy. This document explains how the core algorithm works, based on the paper ["LEANN: A Low-Storage Vector Index"](https://arxiv.org/abs/2506.08276) and the implementation in this repository.

## Table of Contents

- [The Storage Problem](#the-storage-problem)
- [Core Idea: Graph-Based Selective Recomputation](#core-idea-graph-based-selective-recomputation)
- [Key Techniques](#key-techniques)
  - [1. High-Degree Preserving Graph Pruning](#1-high-degree-preserving-graph-pruning)
  - [2. Selective Embedding Recomputation](#2-selective-embedding-recomputation)
  - [3. Compact Storage Format (CSR)](#3-compact-storage-format-csr)
- [Implementation Details](#implementation-details)
  - [Backend Architecture](#backend-architecture)
  - [Build Process](#build-process)
  - [Search Process](#search-process)
- [Performance Characteristics](#performance-characteristics)
- [Code References](#code-references)

---

## The Storage Problem

Traditional vector databases face a fundamental storage challenge:

```
Traditional Vector DB Storage:
- Graph Structure: ~10-20 MB (for 1M documents)
- Embeddings: ~2-4 GB (1M documents × 768 dimensions × 4 bytes)
- Total: ~2-4 GB

The embeddings dominate storage (95-99% of total size)
```

For personal RAG applications with millions of documents (emails, browser history, chat logs), this storage requirement is prohibitive on consumer laptops.

## Core Idea: Graph-Based Selective Recomputation

LEANN's breakthrough insight is: **We don't need to store all embeddings if we can recompute them on-demand during search.**

### The Key Innovation

1. **Store only the graph structure** (10-100 MB) - the connectivity between nodes
2. **Store the original text** (documents/passages)
3. **Recompute embeddings dynamically** during search - but only for nodes visited during graph traversal

### Why This Works

- Graph-based search algorithms (HNSW, DiskANN) visit only a **small fraction** of nodes (typically <1%) during a search
- Modern embedding models are fast enough to compute embeddings for these few nodes on-the-fly
- GPU acceleration makes embedding computation even faster

---

## Key Techniques

### 1. High-Degree Preserving Graph Pruning

**Location in code:** `packages/leann-backend-hnsw/leann_backend_hnsw/convert_to_csr.py`

#### Why Prune?

After building the initial graph index, we have:
- **Graph structure** (neighbor connections)
- **Embeddings** (vector representations)

Traditional vector DBs keep both. LEANN prunes the embeddings while preserving the graph.

#### How It Works

The graph pruning process:

```python
# From convert_to_csr.py

def prune_hnsw_embeddings(input_filename: str, output_filename: str) -> bool:
    """
    Rewrite an HNSW index while dropping the embedded storage section.
    
    This function:
    1. Reads the full HNSW index (graph + embeddings)
    2. Extracts the graph structure (neighbor links, levels, etc.)
    3. Writes back ONLY the graph structure
    4. Marks storage as NULL (no embeddings stored)
    """
    # Read HNSW structure
    levels_np = read_numpy_vector(f_in, np.int32, "i")
    offsets_np = read_numpy_vector(f_in, np.uint64, "Q")
    neighbors_np = read_numpy_vector(f_in, np.int32, "i")
    
    # Write back without embeddings
    write_compact_format(..., storage_fourcc=NULL_INDEX_FOURCC, storage_data=b"")
```

#### Key Insight: Preserve High-Degree Nodes

The graph structure is critical for search quality. LEANN preserves:
- **Hub nodes** (high-degree nodes that connect many parts of the graph)
- **Neighbor relationships** (which nodes connect to which)
- **Hierarchical levels** (from HNSW's layered structure)

These structural features ensure that even without embeddings, the graph can still guide search effectively.

### 2. Selective Embedding Recomputation

**Location in code:** `packages/leann-backend-hnsw/leann_backend_hnsw/hnsw_backend.py` (search method)

#### The Recomputation Process

During search, LEANN:

1. **Receives a query** and computes its embedding
2. **Traverses the graph** using approximate distances (PQ codes or graph structure)
3. **Dynamically recomputes** embeddings for candidate nodes
4. **Reranks** candidates using exact distances

```python
# From hnsw_backend.py - HNSWSearcher.search()

def search(self, query: np.ndarray, top_k: int, zmq_port: int, 
           recompute_embeddings: bool = True, ...):
    """
    Search with selective recomputation.
    
    Args:
        zmq_port: Port of the embedding server that computes embeddings on-demand
        recompute_embeddings: Whether to fetch fresh embeddings during search
    """
    # Set up embedding server connection for recomputation
    if recompute_embeddings:
        if zmq_port is None:
            raise ValueError("zmq_port required for recomputation")
        self._index.set_zmq_port(zmq_port)
    
    # Configure search parameters
    params = faiss.SearchParametersHNSW()
    params.efSearch = complexity
    params.beam_size = beam_width
    
    # Execute search - C++ code will call embedding server for visited nodes
    self._index.search(
        query.shape[0],
        faiss.swig_ptr(query),
        top_k,
        faiss.swig_ptr(distances),
        faiss.swig_ptr(labels),
        params,
    )
```

#### The Embedding Server Architecture

**Location:** `packages/leann-core/src/leann/embedding_server_manager.py`

LEANN uses a ZMQ-based client-server architecture:

```
┌─────────────────┐          ┌──────────────────┐          ┌─────────────────┐
│   Search        │          │  Embedding       │          │  Embedding      │
│   Process       │  ──────> │  Server          │  ──────> │  Model          │
│   (C++ HNSW)    │  ZMQ     │  (Python)        │          │  (GPU/CPU)      │
└─────────────────┘          └──────────────────┘          └─────────────────┘
      │                              │                              │
      │ Request: ["id1", "id2"]     │ Lookup text for IDs         │
      │ ───────────────────────────>│──────────────────────────>  │
      │                              │ Compute embeddings          │
      │                              │<─────────────────────────── │
      │ Response: embeddings array  │                              │
      │<────────────────────────────│                              │
```

The embedding server:
1. **Receives node IDs** from the search process
2. **Looks up the text** for those IDs from the passage files
3. **Computes embeddings** using the specified model
4. **Returns embeddings** to the search process

#### Batching for Efficiency

```python
# From embedding_compute.py

def compute_embeddings_via_server(chunks: list[str], model_name: str, port: int):
    """
    Compute embeddings using the embedding server.
    Batches multiple chunks for efficient GPU utilization.
    """
    context = zmq.Context()
    socket = zmq.Context().socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    
    # Send batch of text chunks
    socket.send(msgpack.packb(chunks))
    
    # Receive batched embeddings
    embeddings_list = msgpack.unpackb(socket.recv())
    return np.array(embeddings_list, dtype=np.float32)
```

### 3. Compact Storage Format (CSR)

**Location in code:** `packages/leann-backend-hnsw/leann_backend_hnsw/convert_to_csr.py`

#### Why CSR (Compressed Sparse Row)?

The graph structure itself can be optimized. LEANN converts the standard HNSW format to CSR format:

```
Standard Format (List of Lists):
Node 0: [1, 3, 5, 7]
Node 1: [0, 2, 4]
Node 2: [1, 3]
...

CSR Format (Compressed):
offsets:   [0, 4, 7, 9, ...]     # Where each node's neighbors start
neighbors: [1,3,5,7, 0,2,4, 1,3, ...] # Flattened neighbor lists
```

**Storage savings:**
- Standard format: Each node stores neighbor list separately
- CSR format: Single flattened array + offset array
- Reduction: 20-40% for graph structure alone

#### CSR Conversion Process

```python
# From convert_to_csr.py

def convert_hnsw_graph_to_csr(input_filename, output_filename, prune_embeddings=True):
    """
    Convert HNSW graph to CSR format with optional embedding pruning.
    
    This achieves two optimizations:
    1. Compact graph storage (CSR format)
    2. Remove embeddings (if prune_embeddings=True)
    """
    # Read original format
    levels_np = read_numpy_vector(f_in, np.int32, "i")
    offsets_np = read_numpy_vector(f_in, np.uint64, "Q")
    neighbors_np = read_numpy_vector(f_in, np.int32, "i")
    
    # Convert to CSR
    compact_neighbors_data = []
    compact_level_ptr = []
    compact_node_offsets_np = np.zeros(ntotal + 1, dtype=np.uint64)
    
    # Build CSR structure
    for i in range(ntotal):
        node_max_level = levels_np[i] - 1
        
        # For each level of this node
        for level in range(node_max_level + 1):
            compact_level_ptr.append(current_data_idx)
            
            # Get neighbors at this level
            begin = original_offset_start + get_cum_neighbors(cum_nneighbor_per_level_np, level)
            end = original_offset_start + get_cum_neighbors(cum_nneighbor_per_level_np, level + 1)
            
            # Filter valid neighbors (>= 0)
            level_neighbors = neighbors_np[begin:end]
            valid_neighbors = level_neighbors[level_neighbors >= 0]
            
            # Add to compact format
            compact_neighbors_data.extend(valid_neighbors)
            current_data_idx += len(valid_neighbors)
        
        compact_level_ptr.append(current_data_idx)
        compact_node_offsets_np[i] = node_ptr_start_index
    
    # Write compact format (with or without embeddings)
    write_compact_format(f_out, ..., storage_fourcc, storage_data)
```

---

## Implementation Details

### Backend Architecture

LEANN supports two backends, both implementing the same recomputation strategy:

#### 1. HNSW Backend (Default)

**Location:** `packages/leann-backend-hnsw/`

- **Based on:** FAISS's HNSW implementation with custom modifications
- **Best for:** Most use cases, maximum storage savings through full recomputation
- **Graph structure:** Hierarchical Navigable Small World (multi-layer graph)
- **Search complexity:** O(log N) hops with constant time per hop

Key features:
```python
# From hnsw_backend.py

class HNSWBuilder:
    def __init__(self, **kwargs):
        self.is_compact = kwargs.setdefault("is_compact", True)  # Use CSR format
        self.is_recompute = kwargs.setdefault("is_recompute", True)  # Prune embeddings
        self.M = kwargs.setdefault("M", 32)  # Graph degree
        self.efConstruction = kwargs.setdefault("efConstruction", 200)  # Build quality
```

#### 2. DiskANN Backend

**Location:** `packages/leann-backend-diskann/`

- **Based on:** Microsoft's DiskANN (disk-based ANN search)
- **Best for:** Larger datasets (10M+ documents), superior search speed
- **Graph structure:** Vamana graph with Product Quantization
- **Search complexity:** Uses PQ for fast approximate distances, reranks with exact embeddings

Key features:
```python
# From diskann_backend.py

class DiskannBuilder:
    def build(self, data, ids, index_path, **kwargs):
        # Build DiskANN index
        diskannpy.build_disk_float_index(...)
        
        # Auto-partition if recompute is enabled
        if kwargs.get("is_recompute", False):
            from .graph_partition import partition_graph
            partition_graph(index_prefix_path=absolute_index_prefix_path, ...)
```

**Graph Partitioning** (DiskANN-specific optimization):

Location: `packages/leann-backend-diskann/leann_backend_diskann/graph_partition.py`

DiskANN partitions the graph into clusters for better cache locality:
- Nodes in the same cluster are stored contiguously
- During search, accessing clustered nodes is faster (disk/cache friendly)
- Reduces random access patterns

### Build Process

The index building pipeline:

```
1. User adds text chunks
   ↓
2. Compute embeddings (batch processing)
   ├─ Sentence Transformers (default)
   ├─ OpenAI API
   ├─ MLX (Apple Silicon)
   └─ Ollama (local models)
   ↓
3. Build graph index (HNSW or DiskANN)
   ├─ Create neighbor connections
   ├─ Build hierarchical layers
   └─ Store: graph + embeddings
   ↓
4. Convert to compact format (if is_compact=True)
   ├─ Convert to CSR format
   └─ Prune embeddings (if is_recompute=True)
   ↓
5. Write metadata
   ├─ Index configuration
   ├─ Embedding model info
   └─ Passage sources
```

**Code flow:**

```python
# From api.py - LeannBuilder.build_index()

class LeannBuilder:
    def build_index(self, index_path: str):
        # 1. Compute embeddings for all chunks
        embeddings = compute_embeddings(
            texts_to_embed,
            self.embedding_model,
            self.embedding_mode,
            use_server=False,
            is_build=True,
        )
        
        # 2. Build backend index
        builder_instance = self.backend_factory.builder(**backend_kwargs)
        builder_instance.build(embeddings, string_ids, index_path)
        
        # 3. Backend automatically converts to compact format
        # (see HNSWBuilder._convert_to_csr in hnsw_backend.py)
        
        # 4. Write metadata
        meta_data = {
            "version": "1.0",
            "backend_name": self.backend_name,
            "embedding_model": self.embedding_model,
            "backend_kwargs": self.backend_kwargs,
            "passage_sources": [...],
        }
        with open(leann_meta_path, "w") as f:
            json.dump(meta_data, f)
```

### Search Process

The search pipeline with recomputation:

```
1. User query arrives
   ↓
2. Compute query embedding
   ├─ Use embedding server (if available)
   └─ Or direct computation
   ↓
3. Start embedding server (for recomputation)
   ├─ Launch ZMQ server
   ├─ Load embedding model
   └─ Provide access to passage files
   ↓
4. Execute graph search
   ├─ Start at entry point
   ├─ For each candidate node:
   │  ├─ Request embedding (via ZMQ)
   │  ├─ Compute distance
   │  └─ Update candidate set
   └─ Return top-k results
   ↓
5. Enrich results with text
   ├─ Load passage metadata
   └─ Return SearchResults
```

**Code flow:**

```python
# From api.py - LeannSearcher.search()

class LeannSearcher:
    def search(self, query: str, top_k: int, recompute_embeddings: bool = True, ...):
        # 1. Compute query embedding
        query_embedding = self.backend_impl.compute_query_embedding(
            query,
            use_server_if_available=recompute_embeddings,
        )
        
        # 2. Start embedding server (if recomputation enabled)
        if recompute_embeddings:
            zmq_port = self.backend_impl._ensure_server_running(
                self.meta_path_str,
                port=expected_zmq_port,
            )
        
        # 3. Execute search with recomputation
        results = self.backend_impl.search(
            query_embedding,
            top_k,
            zmq_port=zmq_port,
            recompute_embeddings=recompute_embeddings,
            ...
        )
        
        # 4. Enrich with text
        enriched_results = []
        for string_id, dist in zip(results["labels"][0], results["distances"][0]):
            passage_data = self.passage_manager.get_passage(string_id)
            enriched_results.append(
                SearchResult(
                    id=string_id,
                    score=dist,
                    text=passage_data["text"],
                    metadata=passage_data.get("metadata", {}),
                )
            )
        
        return enriched_results
```

---

## Performance Characteristics

### Storage Efficiency

From the README benchmarks:

| System | DPR (2.1M) | Wiki (60M) | Chat (400K) | Email (780K) | Browser (38K) |
|--------|-------------|------------|-------------|--------------|---------------|
| Traditional | 3.8 GB | 201 GB | 1.8 GB | 2.4 GB | 130 MB |
| LEANN  | 324 MB | 6 GB | 64 MB | 79 MB | 6.4 MB |
| **Savings** | **91%** | **97%** | **97%** | **97%** | **95%** |

### Search Speed Trade-offs

**Without recomputation** (baseline):
- Search time: ~5-10ms for 1M documents
- Limited by graph traversal speed

**With recomputation** (LEANN):
- Search time: ~20-50ms for 1M documents
- Additional overhead from:
  - Embedding computation (10-30ms for ~100 nodes)
  - ZMQ communication (~1-2ms)

**Key insight:** The 2-5x slowdown is acceptable for:
- Interactive personal search (still < 100ms)
- Applications where storage is the bottleneck
- Consumer hardware without sufficient RAM/disk

### Embedding Computation Optimization

LEANN optimizes recomputation through:

1. **Batching:** Combine multiple embedding requests
   ```python
   # Instead of: compute_embedding(text1), compute_embedding(text2), ...
   # Do: compute_embeddings([text1, text2, text3, ...])
   ```

2. **GPU acceleration:** Leverage GPU for batch embedding
   ```python
   # Embedding models run on GPU by default
   model = SentenceTransformer(model_name, device='cuda')
   embeddings = model.encode(texts, batch_size=32)
   ```

3. **Smart candidate selection:** Visit only promising nodes
   - HNSW's beam search focuses on high-quality candidates
   - DiskANN's PQ helps skip irrelevant regions

---

## Code References

### Core Algorithm Files

1. **Graph Pruning:**
   - `packages/leann-backend-hnsw/leann_backend_hnsw/convert_to_csr.py`
     - `convert_hnsw_graph_to_csr()`: Main conversion function
     - `prune_hnsw_embeddings()`: Embedding removal
     - `write_compact_format()`: CSR format writer

2. **Search with Recomputation:**
   - `packages/leann-backend-hnsw/leann_backend_hnsw/hnsw_backend.py`
     - `HNSWSearcher.search()`: Search implementation
     - ZMQ port configuration for embedding server
   - `packages/leann-backend-diskann/leann_backend_diskann/diskann_backend.py`
     - `DiskannSearcher.search()`: DiskANN search with recomputation

3. **Embedding Server:**
   - `packages/leann-core/src/leann/embedding_server_manager.py`
     - `EmbeddingServerManager`: Server lifecycle management
     - `start_server()`: Launch embedding server
   - `packages/leann-core/src/leann/embedding_compute.py`
     - `compute_embeddings()`: Direct embedding computation
     - `compute_embeddings_via_server()`: ZMQ-based computation

4. **API Layer:**
   - `packages/leann-core/src/leann/api.py`
     - `LeannBuilder`: Index construction
     - `LeannSearcher`: Search interface
     - `PassageManager`: Text storage and retrieval

5. **Backend Interfaces:**
   - `packages/leann-core/src/leann/interface.py`
     - `LeannBackendFactoryInterface`: Backend abstraction
     - `LeannBackendBuilderInterface`: Build interface
     - `LeannBackendSearcherInterface`: Search interface

### Graph Algorithms (C++ Extensions)

The actual graph traversal happens in C++/CUDA:

- **HNSW:** Modified FAISS library (included as submodule)
  - Graph construction: HNSW's hierarchical layer building
  - Search: Beam search with configurable `efSearch` and `beam_size`
  - Recomputation hooks: ZMQ calls during search

- **DiskANN:** Microsoft DiskANN library (included as submodule)
  - Graph construction: Vamana graph building
  - Product Quantization: For fast approximate distances
  - Disk-based storage: Optimized for SSD access patterns
  - Graph partitioning: Clustering for cache locality

---

## Summary

LEANN achieves dramatic storage reduction through three key innovations:

1. **Graph-based selective recomputation:**
   - Store only graph structure, recompute embeddings on-demand
   - Visit <1% of nodes during search, making recomputation practical

2. **High-degree preserving pruning:**
   - Remove embeddings but preserve graph topology
   - Keep hub nodes and connectivity patterns for search quality

3. **Compact storage format:**
   - CSR format for efficient graph storage
   - Optimized serialization reduces graph overhead

The result: **97% storage reduction** with only 2-5x slower search, enabling personal RAG systems with millions of documents on consumer laptops.

---

## References

- Paper: [LEANN: A Low-Storage Vector Index](https://arxiv.org/abs/2506.08276)
- FAISS: [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- DiskANN: [Microsoft DiskANN](https://github.com/microsoft/DiskANN)
- HNSW Algorithm: [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)
