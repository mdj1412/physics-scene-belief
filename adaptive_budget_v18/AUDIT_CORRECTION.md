# Independent audit seed correction

Primary master91531 uses representative seed92531. Original alternative master92531 also uses outer stream92531. Despite separate dimensions, calling the two whole runs independent is not justified by distinct seeds. This was found by code review before inspecting decision-loss results. Primary policies, thresholds, data and all original audits are preserved unchanged.

Replace ONLY the secondary acquisition-repeatability and isolated-outer-perturbation audits by master101531 (representative102531, outer101531/101631/101731/101831), disjoint from primary representative92531 and outer91531/91631/91731/91831 and reference93531/93631/93731/93831. Original seed92531 audit files remain in results/audits and are not used for the final independent-repeatability claims.

The supplemental script reads public factual/context and primary stopping K/m only, never private outcomes. Its corrected policy output is separately frozen before it is used in final loss analysis. Equivalent optimized kernels are verified against all primary policies first. The change is an independence correction, not an outcome-based seed selection.
