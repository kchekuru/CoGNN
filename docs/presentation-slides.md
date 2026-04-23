# Cooperative Graph Neural Networks — Presentation Slides
**Paper:** [Cooperative Graph Neural Networks](https://arxiv.org/abs/2310.01267) · ICML 2024

---

## Slide 1 — Motivation & Problem Statement

**Title:** Why Standard GNNs Fall Short

**Key Points:**
- GNNs aggregate messages from *all* neighbours at every layer — a fixed, non-selective strategy.
- Real graphs contain **noisy, task-irrelevant, or heterophilic edges** that corrupt node representations.
- Two central failure modes:
  - **Over-smoothing** — deep aggregation washes out discriminative features.
  - **Over-squashing** — long-range signals get compressed through bottleneck edges.
- Existing fixes (DropEdge, graph rewiring, attention) either act statically, require pre-processing, or lack a principled communication model.
- **Key question:** Can nodes *learn* which edges to use — and do so *collaboratively*, layer by layer?

---

## Slide 2 — Related Work

**Title:** Prior Approaches to Adaptive Message Passing

| Approach | Idea | Limitation |
|---|---|---|
| Graph Attention (GAT) | Soft edge weights via attention | Weights are input-dependent but not discrete; all edges always contribute |
| DropEdge | Random edge removal during training | Not task-adaptive; purely a regulariser |
| Graph Rewiring (DIGL, SDRF) | Static structural pre-processing | Fixed before training; no layer-wise adaptation |
| Stochastic Depth GNNs | Random layer/edge dropping | Not conditioned on learned representations |
| IGNN / NerveNet | Agent/environment separation | Domain-specific; not general GNN frameworks |

**Gap:** No method allows each node to *dynamically and jointly* decide, per layer, whether to receive and send messages — framed as a cooperative multi-agent decision.

---

## Slide 3 — Methodology: The CoGNN Framework

**Title:** Cooperative Agents on a Graph

**Core idea:** Treat each node as an agent that independently decides whether to **send** and **receive** messages at each GNN layer.

**Architecture — two interleaved networks per layer:**

1. **Action Network (`ActionNet`)** — a small GNN that, for each node, outputs logits over {pass, block} for both *in-edges* and *out-edges*.
2. **Environment Network (`EnvNet`)** — the main GNN (GCN, GIN, etc.) that performs message passing, masked by the action decisions.

**Edge weight construction (per layer):**
$$w_{(u,v)} = \text{keep\_out}(u) \times \text{keep\_in}(v)$$
An edge is active only when the sender *wants to send* **and** the receiver *wants to receive* — cooperation.

**Discrete decisions via Gumbel-Softmax:**
- Hard, differentiable sampling during training (`tau` is fixed or learned via `TempSoftPlus`).
- Allows end-to-end training despite discrete edge on/off decisions.

**Skip connections, layer norm, and dropout** are supported for stability.

---

## Slide 4 — Experiments & Results: Node Classification

**Title:** Heterophilic & Homophilic Node Classification

**Datasets:**
- Heterophilic: `roman_empire`, `amazon_ratings`, `minesweeper`, `tolokers`, `questions` (from PyG)
- Homophilic: `cora`, `pubmed` (Planetoid splits)

**Evaluation Protocol:** 10-fold cross-validation; metric = accuracy (or ROC-AUC for binary tasks).

**Key Results (from paper, Table 1 & 2):**

| Dataset | Best Baseline | CoGNN | Δ |
|---|---|---|---|
| Roman Empire | ~88% | **~90%** | +2% |
| Amazon Ratings | ~49% | **~53%** | +4% |
| Minesweeper | ~97% | **~98%** | +1% |
| Cora | ~87% | **~88%** | +1% |

- CoGNN consistently outperforms or matches GCN, GIN, GCNII, and GAT on both homophilic and heterophilic graphs.
- Gains are largest on **heterophilic** datasets where noisy neighbour filtering matters most.

---

## Slide 5 — Experiments & Results: Graph Classification & LRGB

**Title:** Graph-Level Tasks and Long-Range Benchmarks

**Datasets:**
- TU Datasets: `IMDB-B`, `IMDB-M`, `RDT-B`, `RDT-M`, `ENZYMES`, `PROTEINS`, `NCI1`
- LRGB: `Peptides-func` (multi-label, AP metric)

**Evaluation Protocol:** 10-fold CV for TU; official splits for LRGB. Pooling: mean or sum.

**Key Results (from paper):**

| Dataset | GIN (baseline) | CoGNN | Δ |
|---|---|---|---|
| IMDB-B | 74.4% | **76.2%** | +1.8% |
| RDT-B | 89.9% | **91.5%** | +1.6% |
| Peptides-func (AP) | 0.628 | **0.647** | +0.019 |

- CoGNN improves over base GNN (GCN/GIN) backbones without architectural surgery.
- On Peptides-func, CoGNN's selective long-range communication helps capture functional motifs across the molecule.

---

## Slide 6 — Experiments & Results: Ablation & Analysis

**Title:** What Actually Drives the Gains?

**Ablation Studies (from paper):**

| Variant | Performance |
|---|---|
| Full CoGNN | Best |
| Fixed temperature (no `learn_temp`) | Slight drop |
| Remove out-action (only in-action) | Moderate drop |
| Remove in-action (only out-action) | Similar moderate drop |
| Both actions removed (= base GNN) | Largest drop |

**Takeaways:**
- Both in- and out-actions are necessary — cooperation is the key, not just filtering in one direction.
- Learnable temperature (`TempSoftPlus`) helps but is not the primary contributor.
- **Edge ratio analysis:** CoGNN learns to use ~40–70% of edges depending on the layer and dataset, showing genuine selective pruning, not random dropout.
- Synthetic experiments (`cycles`, `root_neighbours`) confirm CoGNN can learn *exact* structural patterns that naive aggregation cannot distinguish.

---

## Slide 7 — Critical Analysis

**Title:** Strengths, Weaknesses & Applicability

**Strengths:**
- Elegant framing: cooperative MARL meets GNNs — principled and general.
- Drop-in compatible: any GNN backbone (GCN, GIN) can serve as the environment network.
- Jointly learned structure and features — no separate pre-processing step.
- Scales to node and graph classification, homophilic and heterophilic settings.

**Weaknesses / Limitations:**
- **Computational overhead:** doubles the number of GNN forward passes per layer (ActionNet + EnvNet).
- **Gumbel-Softmax approximation:** hard samples are non-differentiable at test time; behaviour can differ between train and inference.
- **Hyperparameter sensitivity:** temperature `tau0`, action network depth, and environment depth all interact — tuning cost is higher.
- **Discrete actions are binary:** nodes either fully keep or fully drop edges; soft weighting (as in GAT) may be more expressive in some settings.
- Results on very large graphs (OGB-scale) are not reported in the paper.

---

## Slide 8 — Future Work & Discussion

**Title:** Open Directions

**From the paper:**
- Extending CoGNN to **directed and heterogeneous graphs**.
- Applying to **dynamic graphs** where the action policy can adapt over time.

**Beyond the paper:**
- Replace Gumbel-Softmax with learned **straight-through estimators** or reinforcement learning rewards tied to downstream task performance.
- Combine CoGNN's selective communication with **positional encodings** (Laplacian, RWSE) for long-range graph transformers.
- Explore **edge-feature-conditioned** actions (already partially supported via `act_bond_encoder` in the code).
- Scale to **OGB / large-scale benchmarks** with mini-batch neighbourhood sampling.

**Discussion Questions:**
1. When is cooperative pruning more useful than soft attention weighting?
2. How does the binary action interact with over-squashing — can it make it worse on some graph topologies?
3. Is the Gumbel temperature the right inductive bias, or would a learned discrete prior work better?
