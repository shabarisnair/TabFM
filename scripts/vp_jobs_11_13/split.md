| script | jobs | GPU-hours (est., H100) | sigmoid / m1 jobs (gas-drift, coil2000) |
|---|---|---|---|
| gpu_00.sh | 45 | 0.25 | 0 |
| gpu_01.sh | 45 | 0.25 | 0 |
| gpu_02.sh | 45 | 0.25 | 0 |
| gpu_03.sh | 46 | 0.25 | 0 |
| gpu_04.sh | 46 | 0.25 | 0 |
| gpu_05.sh | 46 | 0.25 | 0 |
| gpu_06.sh | 43 | 0.25 | 0 |
| gpu_07.sh | 43 | 0.25 | 0 |
| gpu_08.sh | 43 | 0.25 | 0 |
| gpu_09.sh | 46 | 0.25 | 0 |
| gpu_10.sh | 46 | 0.25 | 0 |
| gpu_11.sh | 46 | 0.25 | 0 |

540 jobs, about 3.0 H100-hours in all, 0.25 on the busiest script (on a GPU that another busy job shares, two to four times that). Each script needs about 4 GiB free: these jobs are TabPFN inference only (gas-drift peaks at 2.6 GiB).
