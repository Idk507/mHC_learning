```markdown
# Manifold-Constrained Hyper-Connections (mHC) – Numerical Walkthrough

This document provides a **step-by-step, matrix-based numerical example** of the forward pass through a **Manifold-Constrained Hyper-Connection (mHC)** layer.

Goal: Make the full computation transparent using small dimensions and concrete numbers.

## Setup

- Hidden dimension **C = 2**
- Number of streams **S = 3**
- Batch size **B = 1** (for clarity)

**Input vector**

```math
\mathbf{x} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}
```

## Step 1: Stream Expansion

Replicate the input into **S** identical streams:

```math
\mathbf{X} = \begin{bmatrix}
1.0 & 2.0 \\
1.0 & 2.0 \\
1.0 & 2.0
\end{bmatrix} \in \mathbb{R}^{3 \times 2}
```

## Step 2: Read Vector (pre-softmax attention weights)

Read logits:

```math
\mathbf{r} = \begin{bmatrix} 0.2 & 0.5 & 0.3 \end{bmatrix}
```

```math
\mathbf{h}^{\text{pre}} = \mathrm{softmax}(\mathbf{r}) \approx \begin{bmatrix} 0.289 & 0.391 & 0.320 \end{bmatrix}
```

*(Slight difference from the original post due to more precise computation; we'll use rounded values for readability)*

## Step 3: Aggregate Message Input (weighted sum)

```math
\mathbf{u} = \sum_{i=1}^{3} h^{\text{pre}}_i \cdot \mathbf{X}_i
```

Because all rows of **X** are identical:

```math
\mathbf{u} = (0.289 + 0.391 + 0.320) \cdot \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}
```

## Step 4: Layer Function F(u)

For illustration we use the identity transformation:

```math
\mathbf{W} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad F(\mathbf{u}) = \mathbf{Wu} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}
```

## Step 5: Write Vector (post-message weights)

Write logits:

```math
\mathbf{w} = \begin{bmatrix} 0.4 & 0.3 & 0.3 \end{bmatrix}
```

```math
\mathbf{h}^{\text{post}} = \mathrm{softmax}(\mathbf{w}) \approx \begin{bmatrix} 0.356 & 0.322 & 0.322 \end{bmatrix}
```

## Step 6: Expand Message to All Streams

```math
\mathbf{Y}_i = h^{\text{post}}_i \cdot F(\mathbf{u})
```

```math
\mathbf{Y} \approx \begin{bmatrix}
0.356 & 0.712 \\
0.322 & 0.644 \\
0.322 & 0.644
\end{bmatrix}
```

## Step 7: Residual Mixing Matrix (Wᵣ) via Sinkhorn-normalized exp(A)

Residual logits matrix **A**:

```math
\mathbf{A} = \begin{bmatrix}
1 & 2 & 1 \\
1 & 3 & 1 \\
2 & 1 & 2
\end{bmatrix}
```

```math
\mathbf{M} = \exp(\mathbf{A}) \approx \begin{bmatrix}
2.718 & 7.389 & 2.718 \\
2.718 & 20.086 & 2.718 \\
7.389 & 2.718 & 7.389
\end{bmatrix}
```

After Sinkhorn normalization (row & column sums ≈ 1), the example approximates:

```math
\mathbf{W}_r \approx \begin{bmatrix}
0.30 & 0.40 & 0.30 \\
0.25 & 0.50 & 0.25 \\
0.45 & 0.10 & 0.45
\end{bmatrix}
```

*(Note: exact values depend on number of Sinkhorn iterations)*

## Step 8: Mix Residual Streams

```math
\text{Residual update} = \mathbf{W}_r \mathbf{X}
```

Because all rows of **X** are identical → every row of the result is still `[1.0, 2.0]`

## Step 9: Combine Message + Residual

```math
\mathbf{X}' = \mathbf{W}_r \mathbf{X} + \mathbf{Y} \approx \begin{bmatrix}
1.000 + 0.356 & 2.000 + 0.712 \\
1.000 + 0.322 & 2.000 + 0.644 \\
1.000 + 0.322 & 2.000 + 0.644
\end{bmatrix}
= \begin{bmatrix}
1.356 & 2.712 \\
1.322 & 2.644 \\
1.322 & 2.644
\end{bmatrix}
```

## Step 10: Collapse Back to Single Output (using read weights again)

```math
\mathbf{x}' = \sum_{i=1}^{3} h^{\text{pre}}_i \cdot \mathbf{X}'_i
```

Approximate computation:

- dim 1:  
  0.289·1.356 + 0.391·1.322 + 0.320·1.322 ≈ **1.330**

- dim 2:  
  0.289·2.712 + 0.391·2.644 + 0.320·2.644 ≈ **2.662**

```math
\mathbf{x}' \approx \begin{bmatrix} 1.33 & 2.66 \end{bmatrix}
```

## Summary

Input  
→ `x = [1.0, 2.0]`

After mHC (multi-stream message + manifold-constrained residual mixing + recombination)  
→ `x' ≈ [1.33, 2.66]`

The update comes almost entirely from the **message path** (Y), while the **residual path** remains very close to identity due to the Sinkhorn constraint — this is the core idea behind manifold-constrained hyper-connections.

## Next Steps / Extensions

- Derive the **backward pass** (especially through Sinkhorn)
- Gradient flow through softmax(read) and softmax(write)
- Non-identity F(·) (gated MLP, attention, etc.)
- Larger S and C values
- Actual Sinkhorn implementation in code

Feel free to open issues / PRs with improvements, more precise numbers, or PyTorch/JAX implementations!
```

You can copy the content above into a file named `README.md`.

Let me know if you'd like:

- more precise numerical values throughout  
- a PyTorch-like pseudocode version  
- LaTeX → GitHub-flavored math adjustments  
- diagrams ( mermaid / SVG placeholders )  

Happy to refine! 🚀
