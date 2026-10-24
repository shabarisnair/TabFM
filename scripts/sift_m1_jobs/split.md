| script | jobs | GPU-hours (est., H100) | jobs |
|---|---|---|---|
| gpu_00.sh | 4 | 0.58 | gas-drift_R30_c1_s0 tabsift_m1, heloc_R30_c3_s0 tabsift_m1, ecommerce_shipping_R30_c3_s0 tabsift_m1, is_this_a_good_customer_R30_c5_s0 tabsift_m1 |
| gpu_01.sh | 3 | 0.55 | gas-drift_R30_c3_s0 tabsift_m1, heloc_R30_c5_s0 tabsift_m1, ecommerce_shipping_R30_c5_s0 tabsift_m1 |
| gpu_02.sh | 4 | 0.57 | gas-drift_R30_c5_s0 tabsift_m1, students_dropout_R30_c1_s0 tabsift_m1, students_dropout_R30_c5_s0 tabsift_m1, is_this_a_good_customer_R30_c3_s0 tabsift_m1 |
| gpu_03.sh | 2 | 0.59 | coil2000_insurance_policies_R30_c1_s0 tabsift_m1, coil2000_insurance_policies_R30_c5_s0 tabsift_m1 |
| gpu_04.sh | 5 | 0.57 | coil2000_insurance_policies_R30_c3_s0 tabsift_m1, heloc_R30_c1_s0 tabsift_m1, students_dropout_R30_c3_s0 tabsift_m1, ecommerce_shipping_R30_c1_s0 tabsift_m1, is_this_a_good_customer_R30_c1_s0 tabsift_m1 |

18 jobs, about 2.9 H100-hours in all, 0.59 on the busiest script (on a GPU that another busy job shares, two to four times that).
