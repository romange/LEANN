# LEANN Core Algorithm: Quick Reference

This document provides a quick reference guide to understanding the LEANN algorithm and navigating the implementation.

## 🎯 Quick Start: Understanding LEANN

**New to LEANN?** Start here:

1. **Visual Overview:** [How It Works (Visual Guide)](./HOW_IT_WORKS.md)
   - High-level diagrams and illustrations
   - Easy to understand for non-technical users
   - Shows the storage problem and LEANN's solution

2. **Technical Deep Dive:** [Algorithm Explanation (Technical)](./ALGORITHM_EXPLANATION.md)
   - Detailed algorithm description
   - Code references and implementation details
   - For developers and researchers

3. **Research Paper:** [LEANN: A Low-Storage Vector Index](https://arxiv.org/abs/2506.08276)
   - Formal algorithmic analysis
   - Benchmark results and comparisons
   - Theoretical foundations

## 🔑 Key Concepts

### The Core Idea

Traditional vector databases store:
```
Graph (10 MB) + Embeddings (2 GB) = 2.01 GB
```

LEANN stores:
```
Graph (10 MB) + Text (50 MB) = 60 MB
```

**How?** Recompute embeddings on-demand during search!

### Three Key Techniques

1. **Graph-Based Selective Recomputation**
   - Only compute embeddings for ~100 nodes visited during search
   - Saves 99% of embedding computations
   - See: [Algorithm Explanation - Selective Recomputation](./ALGORITHM_EXPLANATION.md#2-selective-embedding-recomputation)

2. **High-Degree Preserving Pruning**
   - Remove embeddings but keep graph structure
   - Preserve "hub" nodes for search quality
   - See: [Algorithm Explanation - Graph Pruning](./ALGORITHM_EXPLANATION.md#1-high-degree-preserving-graph-pruning)

3. **Compact Storage Format (CSR)**
   - Compress graph structure by 20-40%
   - Efficient memory access patterns
   - See: [Algorithm Explanation - CSR Format](./ALGORITHM_EXPLANATION.md#3-compact-storage-format-csr)

## 📂 Code Structure

### Core Implementation Files

```
packages/
├── leann-backend-hnsw/          # HNSW backend (default)
│   ├── hnsw_backend.py          # Main backend implementation
│   ├── convert_to_csr.py        # Graph pruning and CSR conversion
│   └── hnsw_embedding_server.py # Embedding server for HNSW
│
├── leann-backend-diskann/       # DiskANN backend (advanced)
│   ├── diskann_backend.py       # DiskANN implementation
│   └── graph_partition.py       # Graph partitioning for disk locality
│
└── leann-core/src/leann/        # Core API
    ├── api.py                   # LeannBuilder and LeannSearcher
    ├── embedding_compute.py     # Embedding computation
    └── embedding_server_manager.py # Server lifecycle management
```

### Key Functions

#### Building an Index

```python
# File: packages/leann-core/src/leann/api.py
# Class: LeannBuilder

def build_index(self, index_path: str):
    """
    1. Compute all embeddings (batch processing)
    2. Build graph index (HNSW or DiskANN)
    3. Prune embeddings (convert to compact format)
    4. Save metadata
    """
```

See: [Algorithm Explanation - Build Process](./ALGORITHM_EXPLANATION.md#build-process)

#### Searching with Recomputation

```python
# File: packages/leann-backend-hnsw/leann_backend_hnsw/hnsw_backend.py
# Class: HNSWSearcher

def search(self, query, top_k, zmq_port, recompute_embeddings=True):
    """
    1. Start embedding server (for recomputation)
    2. Traverse graph with selective recomputation
    3. Return top-k results
    """
```

See: [Algorithm Explanation - Search Process](./ALGORITHM_EXPLANATION.md#search-process)

#### Graph Pruning

```python
# File: packages/leann-backend-hnsw/leann_backend_hnsw/convert_to_csr.py

def convert_hnsw_graph_to_csr(input_filename, output_filename, prune_embeddings=True):
    """
    1. Read HNSW index (graph + embeddings)
    2. Convert to CSR format
    3. Optionally prune embeddings
    4. Write compact index
    """
```

See: [Algorithm Explanation - CSR Conversion](./ALGORITHM_EXPLANATION.md#3-compact-storage-format-csr)

## 🎓 Learning Path

### For Users

1. Read the [README Architecture section](../README.md#-architecture--how-it-works)
2. Browse the [Visual Guide](./HOW_IT_WORKS.md)
3. Try the [Quick Start examples](../README.md#quick-start)

### For Developers

1. Read the [Visual Guide](./HOW_IT_WORKS.md) for concepts
2. Study the [Algorithm Explanation](./ALGORITHM_EXPLANATION.md) for details
3. Explore the code:
   - Start with `packages/leann-core/src/leann/api.py` (high-level API)
   - Then `packages/leann-backend-hnsw/leann_backend_hnsw/hnsw_backend.py` (backend)
   - Finally `packages/leann-backend-hnsw/leann_backend_hnsw/convert_to_csr.py` (graph pruning)

### For Researchers

1. Read the [research paper](https://arxiv.org/abs/2506.08276)
2. Review the [Algorithm Explanation](./ALGORITHM_EXPLANATION.md)
3. Study the C++ graph traversal code in the FAISS/DiskANN submodules
4. Run the [benchmarks](../benchmarks/) to reproduce results

## 🔬 Advanced Topics

### Backend Comparison

| Feature | HNSW Backend | DiskANN Backend |
|---------|--------------|-----------------|
| **Best for** | Most use cases | Large datasets (10M+) |
| **Storage** | Minimal (CSR + pruned) | Minimal (partitioned) |
| **Search speed** | Fast (10-50ms) | Faster (5-30ms) |
| **Build time** | Medium | Longer |
| **Memory usage** | Low | Very low (disk-based) |
| **Complexity** | Simple | Advanced |

See: [Algorithm Explanation - Backend Architecture](./ALGORITHM_EXPLANATION.md#backend-architecture)

### Performance Tuning

**Build parameters:**
- `M` (graph degree): Higher = better quality, more storage (default: 32)
- `efConstruction`: Higher = better quality, slower build (default: 200)
- `is_compact`: Use CSR format (default: True)
- `is_recompute`: Prune embeddings (default: True)

**Search parameters:**
- `complexity` (efSearch): Higher = better accuracy, slower (default: 64)
- `beam_width`: Parallel search paths (default: 1)
- `recompute_embeddings`: Enable on-demand embedding (default: True)

See: [Configuration Guide](./configuration-guide.md)

### Embedding Server Architecture

```
┌──────────────┐    ZMQ     ┌─────────────────┐    Model    ┌──────────┐
│   Search     │ ────────> │ Embedding Server │ ─────────> │   GPU    │
│   Process    │ <──────── │   (Python)       │ <───────── │          │
│   (C++)      │           └─────────────────┘             └──────────┘
└──────────────┘
```

The embedding server:
- Runs in a separate Python process
- Communicates via ZMQ (inter-process messaging)
- Loads the embedding model once (efficient)
- Batches requests for GPU efficiency

See: [Algorithm Explanation - Embedding Server](./ALGORITHM_EXPLANATION.md#the-embedding-server-architecture)

## 📊 Benchmarks

### Storage Savings

| Dataset | Size | Traditional | LEANN | Savings |
|---------|------|-------------|-------|---------|
| Wikipedia (60M) | Large | 201 GB | 6 GB | 97% |
| Email (780K) | Medium | 2.4 GB | 79 MB | 97% |
| Browser (38K) | Small | 130 MB | 6.4 MB | 95% |

### Search Speed

```
Traditional:  ████░░░░░░ (10ms)
LEANN:        ████████░░ (30ms)

2-3x slower, but still interactive (<100ms)
```

See: [Algorithm Explanation - Performance](./ALGORITHM_EXPLANATION.md#performance-characteristics)

## 🤝 Contributing

Want to improve LEANN? Here's how:

1. **Understand the algorithm:** Read these docs first
2. **Find your area:** Pick a component to enhance
   - Embedding computation: `packages/leann-core/src/leann/embedding_compute.py`
   - Graph algorithms: `packages/leann-backend-hnsw/` or `packages/leann-backend-diskann/`
   - API/UX: `packages/leann-core/src/leann/api.py`
3. **Follow guidelines:** See [CONTRIBUTING.md](./CONTRIBUTING.md)

## 📚 Additional Resources

- **Features:** [features.md](./features.md)
- **FAQ:** [faq.md](./faq.md)
- **Roadmap:** [roadmap.md](./roadmap.md)
- **Configuration Guide:** [configuration-guide.md](./configuration-guide.md)

## 📧 Get Help

- **Slack:** [Join the community](https://join.slack.com/t/leann-e2u9779/shared_invite/zt-3ckd2f6w1-OX08~NN4gkWhh10PRVBj1Q)
- **GitHub Issues:** [Report bugs or ask questions](https://github.com/yichuan-w/LEANN/issues)
- **Research Questions:** See the [paper](https://arxiv.org/abs/2506.08276) and cite it

## 🎉 Summary

LEANN achieves **97% storage reduction** through:

1. **Graph-based selective recomputation** - Only compute embeddings for ~0.01% of nodes
2. **High-degree preserving pruning** - Keep graph structure, remove embeddings
3. **Compact storage format** - CSR compression for efficient graph storage

**Result:** Personal RAG systems with millions of documents on consumer laptops!

---

**Next steps:**
- Read the [Visual Guide](./HOW_IT_WORKS.md) to understand how it works
- Read the [Technical Explanation](./ALGORITHM_EXPLANATION.md) for implementation details
- Try the [examples](../examples/) to see it in action
