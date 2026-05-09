# Algorithmic Complexity Analysis

## Summary
| Metric | Value |
|--------|-------|
| Total Parameters | 34,488,075 |
| Trainable Parameters | 34,488,075 |
| Model Size (FP32) | 131.56 MB |
| Theoretical FLOPs | 15.76 GFLOPs |
| TCN Receptive Field | 85 timesteps |
| Sequence Length | 4096 |
| Fusion Token Dim | 256 |
| Attention Heads | 4 |
| Transformer Layers | 2 |

## Per-Modality Breakdown
| Modality | Params | GFLOPs |
|----------|--------|--------|
| Thermal CNN (EfficientNet-B0) | 5,100,000 | 0.400 |
| Visual CNN (ResNet-50 × 3 views) | 25,600,000 | 12.300 |
| TCN (Causal dilated conv) | 1,200,000 | 0.200 |
| Tabular MLP (3 layers) | 200,000 | 0.001 |
| Cross-modal Fusion Transformer | 2,100,000 | 0.010 |
| Defect + DANN heads | 100,000 | 0.001 |
| **Total (theoretical)** | 34,300,000 | 12.912 |

## Asymptotic Complexity (Big-O)
- **Thermal CNN**: O(H·W·C_in·C_out) where H=W=224
- **Visual CNN**: O(3 × H·W·C_in·C_out) per part (3 sections)
- **TCN**: O(L·C²·K) where L=4096, kernel K=3
- **Fusion Transformer**: O(N²·D) where N=4 tokens, D=256
- **Overall**: O(W·H·C² + L·C²·K + D²)

## Memory Complexity
- Embedding buffer: 4 × 256 = 4KB per sample
- Transformer KV cache: 2 × 2 × 4 × 256 × 4 = negligible
- Full precision model weights: ~131.6 MB