| script | jobs | H100-hours (est.) | longest job | jobs per method |
|---|---|---|---|---|
| gpu_00.sh | 99 | 9.0 | gas-drift_R10_V50_s0 nan_soft (17 min) | nan_soft 14, nan_topk_r* 48, attn_topk 10, attn_soft 14, refs 13 |
| gpu_01.sh | 100 | 9.0 | gas-drift_R20_V50_s0 attn_topk (17 min) | attn_topk 13, nan_topk_r* 52, attn_soft 9, nan_soft 16, refs 10 |
| gpu_02.sh | 104 | 9.0 | gas-drift_R10_V50_s0 attn_topk (17 min) | attn_topk 13, attn_soft 11, nan_soft 14, nan_topk_r* 56, refs 10 |
| gpu_03.sh | 102 | 9.0 | gas-drift_R10_V100_s0 nan_topk_r0 (17 min) | nan_topk_r* 56, attn_topk 12, nan_soft 14, attn_soft 10, refs 10 |
| gpu_04.sh | 108 | 9.0 | gas-drift_R20_V100_s0 attn_soft (17 min) | attn_soft 14, nan_topk_r* 60, attn_topk 8, nan_soft 12, refs 14 |
| gpu_05.sh | 99 | 9.0 | gas-drift_R20_V100_s0 nan_soft (17 min) | nan_soft 12, nan_topk_r* 52, attn_soft 14, attn_topk 11, refs 10 |
| gpu_06.sh | 107 | 9.0 | gas-drift_R30_V100_s0 attn_topk (17 min) | attn_topk 10, nan_topk_r* 60, nan_soft 11, attn_soft 13, refs 13 |
| gpu_07.sh | 97 | 9.0 | gas-drift_R10_V50_s0 attn_soft (17 min) | attn_soft 11, nan_soft 15, attn_topk 13, nan_topk_r* 48, refs 10 |

816 jobs, about 72 H100-hours in all, 9.0 on the busiest script.
