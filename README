# Kernels

# Agent Prompt

```
You have the entire PyTorch codebase indexed. Your goal is to extract the complete CPU implementation of the <opname> operation. This includes:
- The main kernel function(s) executed for the <opname> op on CPU.
- All helper functions or macros it directly or indirectly depends on.
- Any relevant dispatch stubs or registration logic required to understand execution flow.

Important constraints:
- Do not generate or hallucinate any code.
- Only copy actual, verifiable source code from the repository.
- For each code block, prepend a comment with the full path to the source file it was extracted from.

The output should be a single .cpp file containing all this code, structured clearly for downstream use by another agent that will use this as trusted context for kernel rewriting or retargeting.

Goal: Create a self-contained, fully accurate reference of the <opname> CPU kernel implementation and its dependencies.
```




## ops
- add
- sub
- mul
- div
- neg
- rsqrt
- exp
- abs
- log1p
- log
- pow
- min
- clamp_min
- clamp_max
- sigmoid
- nn.function.relu
- tanh
- erf
- lt
- eq
- gt
- ne
- le
- logical_not
- permute
- view
- split
- slice
- cat
- split_with_sizes
- squeeze
- unsequeeze
- var_mean
- sum
- mean
- full
- clone
- slect
- expand
- gather
- amax
- constat_pad_nd
- isnan
- where
- index_sleect
- sdneix_put
- unbind
- bucketize