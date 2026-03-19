# Memory and Launch Path

## Core Mental Model

Every CUDA operator starts with three constraints:
- thread mapping
- memory movement
- synchronization scope

The first design question is not which operator you are implementing, but which access pattern dominates.

## Pattern 1: Element-wise

Examples:
- add
- relu
- sigmoid
- sin

Typical mapping:
- one thread handles one output element
- 1D launch grid
- block size starts from 256

Primary risks:
- non-coalesced reads because of bad layout or stride
- low occupancy from unnecessary registers
- expensive transcendental math on simple kernels

## Pattern 2: Row-wise or Channel-wise

Examples:
- softmax
- layernorm
- reduce-sum along last axis

Typical mapping:
- one block handles one row or one logical reduction unit
- shared memory or warp shuffle for reductions
- often two phases: max/sum or mean/variance

Primary risks:
- reduction overhead
- shared memory bank conflict
- poor utilization for short rows

## Pattern 3: 2D Tile Compute

Examples:
- matmul
- transpose
- convolution tiles

Typical mapping:
- 2D blocks, usually 16x16 or 32x8 or 32x32 depending on registers and smem
- shared memory tiles for reuse
- explicit load/compute/store staging

Primary risks:
- bank conflict in tiles
- register pressure from aggressive unrolling
- incomplete tiles and bounds handling

## Hardware Checklist

Before optimizing, answer these:
- Is the kernel memory-bound or compute-bound?
- Is global memory access coalesced?
- Is shared memory actually increasing reuse?
- Is the launch shape aligned with data layout?
- Is one block doing too little or too much work?

## Minimum Baseline Rule

Every operator should start with a baseline kernel that is:
- obviously correct
- easy to read
- easy to compare against later optimized versions

Do not jump directly to fused or heavily tuned implementations.
