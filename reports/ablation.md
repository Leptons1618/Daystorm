# Modality ablation

Each row forces one modality offline for the whole window by zeroing its
validity mask, exactly as a dead sensor would. Deltas are against `full`.

```
run                    exact match    manoeuvre accuracy           numeric mae    hallucination rate
----------------------------------------------------------------------------------------------------
full                   0.000                 1.000                 0.225                 0.000      
no_camera              0.000                 1.000                 0.206 (-0.02)           0.000      
no_can                 0.000                 1.000                 0.663 (+0.44)           0.208 (+0.21)
no_radar               0.000                 1.000                 0.323 (+0.10)           0.000      
no_audio               0.000                 1.000                 0.419 (+0.19)           0.000      
```
