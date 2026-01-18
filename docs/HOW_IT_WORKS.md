# How LEANN Works: A Visual Guide

This document provides a high-level, visual explanation of how LEANN achieves 97% storage reduction for vector databases.

## The Problem

Traditional vector databases must store both:
1. **Graph structure** (which documents are similar) - ~10 MB
2. **Embeddings** (vector representations) - ~2 GB

For 1 million documents with 768-dimensional embeddings:

```
┌─────────────────────────────────────────┐
│ Traditional Vector Database             │
├─────────────────────────────────────────┤
│ Graph (connections):        10 MB  ▓    │
│ Embeddings (vectors):    2,000 MB  █████│
├─────────────────────────────────────────┤
│ Total:                   2,010 MB       │
└─────────────────────────────────────────┘
```

**Problem:** Embeddings dominate storage (99%), making it impossible to store millions of documents on consumer laptops.

## The Solution: Recompute on Demand

LEANN's insight: **Don't store what you can recompute!**

```
┌─────────────────────────────────────────┐
│ LEANN Index                             │
├─────────────────────────────────────────┤
│ Graph (connections):        10 MB  ▓    │
│ Text (original documents): 50 MB   ██   │
│ Embeddings:                 0 MB        │
├─────────────────────────────────────────┤
│ Total:                      60 MB       │
│ Savings:                        97%     │
└─────────────────────────────────────────┘
```

During search, LEANN recomputes embeddings for only the ~100 nodes visited during graph traversal.

## How It Works: Step by Step

### 1. Build Phase

```
Original Text                 Embedding Model              Graph + Embeddings
─────────────                ─────────────────            ───────────────────
"Machine learning"   ──────>  [0.23, -0.41, ...]  ──────> ╭─── Graph ────╮
"Deep neural nets"   ──────>  [-0.15, 0.62, ...]  ──────> │ Connections  │
"AI algorithms"      ──────>  [0.31, -0.28, ...]  ──────> ├──────────────┤
...                  ──────>  ...                  ──────> │ Embeddings   │
                                                           ╰──────────────╯

                              LEANN Build Process
                              ───────────────────

                         ╭────────────────────────╮
                         │  1. Compute all        │
                         │     embeddings         │
                         ╰──────────┬─────────────╯
                                    ↓
                         ╭────────────────────────╮
                         │  2. Build graph index  │
                         │     (HNSW or DiskANN)  │
                         ╰──────────┬─────────────╯
                                    ↓
                         ╭────────────────────────╮
                         │  3. Prune embeddings   │
                         │     (keep only graph)  │
                         ╰──────────┬─────────────╯
                                    ↓
                         ╭────────────────────────╮
                         │  LEANN Index           │
                         │  ├─ Graph structure    │
                         │  └─ Original text      │
                         ╰────────────────────────╯
```

### 2. Search Phase

```
Query: "neural networks"
─────────────────────────

Step 1: Compute Query Embedding
────────────────────────────────
"neural networks" ──> [0.28, -0.35, 0.51, ...]


Step 2: Graph Traversal with Selective Recomputation
─────────────────────────────────────────────────────

               Entry Point
                   ╭●╮
                  ╱   ╲
                 ●     ●
                ╱ ╲   ╱ ╲
               ●   ● ●   ●
              ╱ ╲  │ │  ╱ ╲
             ●   ●─●─●─●   ●
                 
    ┌─────────────────────────────────────────────┐
    │ For each visited node:                      │
    │                                             │
    │ 1. Lookup text: "Machine learning..."      │
    │ 2. Compute embedding: [0.23, -0.41, ...]   │
    │ 3. Compute distance to query                │
    │ 4. Add to candidates if promising           │
    └─────────────────────────────────────────────┘

Only ~100 out of 1,000,000 nodes are visited!
Only ~100 embeddings need to be recomputed!


Step 3: Return Results
──────────────────────
Top-k candidates sorted by distance
```

### 3. The Magic: Embedding Server

LEANN uses a client-server architecture for on-demand embedding computation:

```
┌──────────────────┐         ┌─────────────────┐         ┌──────────────┐
│  Search Process  │         │ Embedding Server│         │   GPU/CPU    │
│  (C++ Backend)   │         │   (Python)      │         │              │
└────────┬─────────┘         └────────┬────────┘         └──────┬───────┘
         │                            │                         │
         │  "Need embeddings for     │                         │
         │   nodes: 42, 158, 923"    │                         │
         ├──────────────────────────>│                         │
         │                            │  "Lookup text for      │
         │                            │   node 42: 'ML is..'"  │
         │                            ├────────────────────────>│
         │                            │                         │
         │                            │  Batch compute         │
         │                            │  embeddings            │
         │                            │<────────────────────────┤
         │  Return embeddings:       │                         │
         │  [[0.23, ...], ...]       │                         │
         │<──────────────────────────┤                         │
         │                            │                         │
```

## Key Techniques

### 1. High-Degree Preserving Pruning

After building the graph, LEANN removes embeddings but keeps the **graph structure**:

```
Before Pruning:                     After Pruning:
──────────────                      ──────────────

Node 0: neighbors=[1,3,5]           Node 0: neighbors=[1,3,5]
        embedding=[0.23, -0.41]             embedding=NONE
                                    
Node 1: neighbors=[0,2,4]           Node 1: neighbors=[0,2,4]
        embedding=[-0.15, 0.62]             embedding=NONE

The graph structure (connections) is preserved!
The embeddings are removed (97% storage savings)!
```

**Why this works:**
- Graph search only needs connectivity to find candidate nodes
- Exact distances are computed only for visited candidates (via recomputation)
- Graph topology ensures good search quality

### 2. Compact Storage Format (CSR)

LEANN further compresses the graph structure using CSR (Compressed Sparse Row) format:

```
Standard Format:                    CSR Format:
───────────────                     ───────────

Node 0: [1, 3, 5, 7]               offsets:   [0, 4, 7, 9, ...]
Node 1: [0, 2, 4]                  neighbors: [1,3,5,7, 0,2,4, 1,3, ...]
Node 2: [1, 3]
...                                 20-40% additional savings!
```

### 3. Backend Strategies

LEANN supports two backends with different trade-offs:

#### HNSW Backend (Default)

```
Multi-layer graph structure:

Level 2:   ●─────●
          ╱│╲   ╱│╲
Level 1:  ● ● ● ● ●
         ╱│╲│╲│╲│╲│╲
Level 0: ●●●●●●●●●●●●

- Maximum storage savings (full recomputation)
- Good for most use cases
- O(log N) search complexity
```

#### DiskANN Backend

```
Vamana graph + Product Quantization:

      ╭──────────────╮
      │ PQ Codes     │  (fast approximate distances)
      │ (compressed) │
      ╰──────┬───────╯
             │
      ╭──────▼───────╮
      │ Disk Graph   │  (partitioned for cache locality)
      ╰──────┬───────╯
             │
      ╭──────▼───────╮
      │ Recompute    │  (exact distances for candidates)
      │ Embeddings   │
      ╰──────────────╯

- Best for large datasets (10M+ documents)
- Superior search speed
- Disk-optimized storage
```

## Performance Characteristics

### Storage Comparison (60M Documents)

```
Traditional Vector DB:          LEANN:

┌──────────────┐               ┌──────────────┐
│              │               │   Graph: 6GB │
│              │               └──────────────┘
│              │
│ Embeddings   │
│   201 GB     │
│              │
│              │
│              │
│              │
│              │
└──────────────┘
```

**Result:** 201 GB → 6 GB (97% reduction)

### Search Speed Trade-off

```
Traditional (No recomputation):
Search Time: ████░░░░░░ (10ms)
             └─ Graph traversal only

LEANN (With recomputation):
Search Time: ████████░░ (30ms)
             └─ Graph traversal + embedding computation

Still fast enough for interactive use! (<100ms)
```

### Why Recomputation is Practical

**Nodes in database:** 1,000,000
**Nodes visited during search:** ~100 (0.01%)
**Embedding computation time:** ~20-30ms for 100 nodes

```
Without LEANN:
- Store: 1,000,000 embeddings (2 GB)
- Search: Fast (10ms)

With LEANN:
- Store: 0 embeddings (0 GB)
- Search: Compute 100 embeddings (30ms total)
- Extra cost: 20ms (acceptable for 97% storage savings!)
```

## Real-World Applications

### Example: Email Search

```
Traditional:                    LEANN:
───────────                     ──────

780K emails                     780K emails
→ 780K embeddings              → 0 embeddings stored
→ 2.4 GB storage               → 79 MB storage

Search:                         Search:
1. Load all embeddings          1. Traverse graph
2. Compute distances            2. Recompute ~100 embeddings
3. Return top results           3. Return top results
```

**Benefits:**
- Fits on any laptop (79 MB vs 2.4 GB)
- Fast enough for interactive use (< 50ms)
- Complete privacy (all local, no cloud)

### Example: Browser History

```
38K visited pages               38K visited pages
→ 130 MB (traditional)         → 6.4 MB (LEANN)
→ 95% savings!
```

## Conclusion

LEANN achieves dramatic storage reduction through:

1. **Selective recomputation:** Only compute embeddings for visited nodes (~0.01%)
2. **Graph preservation:** Keep connectivity structure for search guidance
3. **Compact storage:** CSR format for efficient graph storage

**Result:** 97% storage reduction with only 2-3x slower search, enabling personal RAG systems with millions of documents on consumer laptops.

---

## Learn More

- **Detailed Algorithm:** See [ALGORITHM_EXPLANATION.md](./ALGORITHM_EXPLANATION.md)
- **Paper:** [LEANN: A Low-Storage Vector Index](https://arxiv.org/abs/2506.08276)
- **Code:** Browse the implementation in `packages/leann-backend-hnsw/` and `packages/leann-backend-diskann/`
