# Implementation Plan

1. Build the minimum gym around synthetic codec tokens with the same multi-codebook dependency structure as the paper.
2. Train and evaluate the no-LT parallel baseline.
3. Train and evaluate AR LT at stack factor 2 to test the core claim and frame-stacking efficiency proxy.
4. Train and evaluate MaskGIT LT at stack factor 2 as an additional paper variant if the first run fits the job budget.
5. Package scripts and environment so future agents can rerun or improve the methods.
