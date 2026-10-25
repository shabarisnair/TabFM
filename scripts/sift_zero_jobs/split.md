| script | jobs | GPU-hours (est., H100) | jobs |
|---|---|---|---|
| gpu_00.sh | 2 | 0.42 | students_dropout_R30_c1_s0 (tabsift_zero), gas-drift_R30_c1_s0 (tabsift_zero) |
| gpu_01.sh | 2 | 0.42 | students_dropout_R30_c3_s0 (tabsift_zero), gas-drift_R30_c3_s0 (tabsift_zero) |
| gpu_02.sh | 2 | 0.42 | students_dropout_R30_c5_s0 (tabsift_zero), gas-drift_R30_c5_s0 (tabsift_zero) |
| gpu_03.sh | 4 | 0.44 | is_this_a_good_customer_R30_c1_s0 (tabsift_zero), ecommerce_shipping_R30_c1_s0 (tabsift_zero), heloc_R30_c1_s0 (tabsift_zero), coil2000_insurance_policies_R30_c1_s0 (tabsift_zero) |
| gpu_04.sh | 4 | 0.44 | is_this_a_good_customer_R30_c3_s0 (tabsift_zero), ecommerce_shipping_R30_c3_s0 (tabsift_zero), heloc_R30_c3_s0 (tabsift_zero), coil2000_insurance_policies_R30_c3_s0 (tabsift_zero) |
| gpu_05.sh | 4 | 0.44 | is_this_a_good_customer_R30_c5_s0 (tabsift_zero), ecommerce_shipping_R30_c5_s0 (tabsift_zero), heloc_R30_c5_s0 (tabsift_zero), coil2000_insurance_policies_R30_c5_s0 (tabsift_zero) |

18 jobs, about 2.6 H100-hours in all, 0.44 on the busiest script (on a GPU that another busy job shares, two to four times that).
