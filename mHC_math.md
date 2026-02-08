To understand **Manifold-Constrained Hyper-Connections (mHC)** deeply from a mathematical and matrix-based implementation standpoint, we’ll now walk through the entire computation **step-by-step** with a **concrete numerical example**. We'll cover every transformation mathematically so that the full data flow through the layer is visible.

---

## ✅ Setup

Let’s define the following:

* Hidden dimension ( C = 2 )
* Number of streams ( S = 3 )
* Batch size ( B = 1 ) (for simplicity)
* Input vector:
  [
  x = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}
  ]

---

## ✅ Step 1: Stream Expansion

**Replicate the input into ( S ) streams.**

[
X = \begin{bmatrix}
1.0 & 2.0 \
1.0 & 2.0 \
1.0 & 2.0
\end{bmatrix} \in \mathbb{R}^{3 \times 2}
]

Each row of ( X ) is a copy of the input vector ( x ).

---

## ✅ Step 2: Read Vector ( h^{\text{pre}} ) (Softmax over Read Logits)

Assume the read logits are:

[
r = \begin{bmatrix} 0.2 & 0.5 & 0.3 \end{bmatrix}
]

Apply softmax:

[
h^{\text{pre}} = \text{softmax}(r) = \frac{e^r}{\sum e^r}
= \frac{1}{e^{0.2} + e^{0.5} + e^{0.3}} \cdot \begin{bmatrix} e^{0.2} & e^{0.5} & e^{0.3} \end{bmatrix}
\approx \begin{bmatrix} 0.254 & 0.339 & 0.407 \end{bmatrix}
]

This vector determines **how much each stream contributes to the message input**.

---

## ✅ Step 3: Aggregate Message Input

Compute:
[
u = \sum_{i=1}^3 h^{\text{pre}}_i \cdot X_i
]

[
u = 0.254 \cdot \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} + 0.339 \cdot \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} + 0.407 \cdot \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}
]

[
u = (0.254 + 0.339 + 0.407) \cdot \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} = 1.0 \cdot \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}
]

So,
[
u = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}
]

This shows **weighted averaging still gives the same input**, due to identical streams.

---

## ✅ Step 4: Layer Function ( F(u) )

Assume ( F(u) = W u ), where

[
W = \begin{bmatrix}
1 & 0 \
0 & 1
\end{bmatrix}
\quad\text{(identity)}
]

Then,
[
F(u) = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}
]

---

## ✅ Step 5: Write Vector ( h^{\text{post}} ) (Softmax over Write Logits)

Assume write logits:

[
w = \begin{bmatrix} 0.4 & 0.3 & 0.3 \end{bmatrix}
]

Softmax:

[
h^{\text{post}} = \text{softmax}(w) \approx \begin{bmatrix} 0.354 & 0.323 & 0.323 \end{bmatrix}
]

---

## ✅ Step 6: Expand Message into Streams

[
Y_i = h^{\text{post}}_i \cdot F(u)
]

Compute each stream:

* ( Y_1 = 0.354 \cdot [1.0, 2.0] = [0.354, 0.708] )
* ( Y_2 = 0.323 \cdot [1.0, 2.0] = [0.323, 0.646] )
* ( Y_3 = 0.323 \cdot [1.0, 2.0] = [0.323, 0.646] )

Message matrix:

[
Y = \begin{bmatrix}
0.354 & 0.708 \
0.323 & 0.646 \
0.323 & 0.646
\end{bmatrix}
]

---

## ✅ Step 7: Residual Mixing Matrix ( W_r )

Assume residual logits matrix:

[
A =
\begin{bmatrix}
1 & 2 & 1 \
1 & 3 & 1 \
2 & 1 & 2
\end{bmatrix}
]

Exponentiate:

[
M = e^A =
\begin{bmatrix}
e^1 & e^2 & e^1 \
e^1 & e^3 & e^1 \
e^2 & e^1 & e^2
\end{bmatrix}
\approx
\begin{bmatrix}
2.718 & 7.389 & 2.718 \
2.718 & 20.085 & 2.718 \
7.389 & 2.718 & 7.389
\end{bmatrix}
]

Apply Sinkhorn normalization (10 iterations) → final ( W_r \approx ) (rounded for simplicity):

[
W_r =
\begin{bmatrix}
0.30 & 0.40 & 0.30 \
0.25 & 0.50 & 0.25 \
0.45 & 0.10 & 0.45
\end{bmatrix}
]

---

## ✅ Step 8: Mix Residual Streams

Compute:
[
R \cdot X =
\begin{bmatrix}
0.30 & 0.40 & 0.30 \
0.25 & 0.50 & 0.25 \
0.45 & 0.10 & 0.45
\end{bmatrix}
\cdot
\begin{bmatrix}
1.0 & 2.0 \
1.0 & 2.0 \
1.0 & 2.0
\end{bmatrix}
=============

\text{Each row} = [1.0, 2.0]
]

So,
[
\text{Residual Mixed Matrix} =
\begin{bmatrix}
1.0 & 2.0 \
1.0 & 2.0 \
1.0 & 2.0
\end{bmatrix}
]

---

## ✅ Step 9: Combine Message and Residual

[
X' = W_r \cdot X + Y =
\begin{bmatrix}
1.0 + 0.354 & 2.0 + 0.708 \
1.0 + 0.323 & 2.0 + 0.646 \
1.0 + 0.323 & 2.0 + 0.646
\end{bmatrix}
=============

\begin{bmatrix}
1.354 & 2.708 \
1.323 & 2.646 \
1.323 & 2.646
\end{bmatrix}
]

This is the **updated stream state**.

---

## ✅ Step 10: Collapse Back to Single Output

Use same ( h^{\text{pre}} = [0.254, 0.339, 0.407] )

[
x' = \sum_i h^{\text{pre}}_i \cdot X'_i
= 0.254 \cdot [1.354, 2.708] + 0.339 \cdot [1.323, 2.646] + 0.407 \cdot [1.323, 2.646]
]

Compute:

* First dimension:
  [
  0.254 \cdot 1.354 + 0.339 \cdot 1.323 + 0.407 \cdot 1.323
  \approx 0.344 + 0.448 + 0.538 = 1.33
  ]
* Second dimension:
  [
  0.254 \cdot 2.708 + 0.339 \cdot 2.646 + 0.407 \cdot 2.646
  \approx 0.687 + 0.897 + 1.077 = 2.661
  ]

[
x' = \begin{bmatrix} 1.33 & 2.661 \end{bmatrix}
]

---

## ✅ Conclusion

We started with input:
[
x = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}
]

And after applying:

* Multi-stream replication
* Learned weighted message computation
* Sinkhorn-projected residual mixing
* Stream recombination

We obtained:
[
x' = \begin{bmatrix} 1.33 & 2.66 \end{bmatrix}
]

All intermediate matrices, residual updates, and learnable mixings contributed to this. The manifold constraint ensured the **residual component was stable** and identity-like, while **the message update** added new signal directionally via the write vector.

This breakdown gives you an **entire end-to-end math-only forward pass** for mHC using exact matrix operations. Let me know if you want backward pass derivations, gradient propagation math through Sinkhorn, or any customizations.
