| script | jobs | H100-hours (est.) | longest job | jobs per method |
|---|---|---|---|---|
| gpu_00.sh | 20 | 0.2 | gas-drift_R10_c1_s0 attn_soft (2 min) | attn_topk 8, attn_soft 12 |
| gpu_01.sh | 20 | 0.2 | gas-drift_R10_c3_s0 attn_soft (2 min) | attn_topk 12, attn_soft 8 |
| gpu_02.sh | 23 | 0.2 | gas-drift_R10_c3_s0 attn_topk (2 min) | attn_topk 11, attn_soft 12 |

63 jobs, about 0.6 H100-hours in all.
