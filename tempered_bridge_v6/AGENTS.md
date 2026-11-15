# v6 constraints

- Parent constraints apply. The user's explicit authorization to use the two idle GPUs overrides the initial pilot's CPU-only setting.
- Use only currently idle GPUs within the user's allocations after checking process and memory state. Initially these were 2676323 and 2676324; both later acquired external training and our inference was stopped. Do not touch other processes or jobs. The user's idle-GPU authorization persists: after the previous training processes exit and repeated checks confirm idle status, the same work may resume. Use the runtime resource guard.
- Compute runs on allocated compute nodes. Local work is preparation, analysis, and reporting.
- Preserve v5 outputs. Reuse their factual data and outcome scores explicitly; never expose truth or unselected measurements to acquisition.
- Korean prose, English technical terms. Do not call resampled ESS or one SMC run proof of convergence.
