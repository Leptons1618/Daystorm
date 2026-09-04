# Modality ablation

Each row forces one modality offline for the whole window by zeroing its
validity mask, exactly as a dead sensor would. Deltas are against `full`.

```
run                    exact match    manoeuvre accuracy           numeric mae    hallucination rate
----------------------------------------------------------------------------------------------------
full                   0.000                 1.000                 0.260                 0.021      
no_camera              0.000                 1.000                 0.373 (+0.11)           0.000 (-0.02)
no_can                 0.000                 1.000                 0.454 (+0.19)           0.208 (+0.19)
no_radar               0.000                 1.000                 0.323 (+0.06)           0.021      
no_audio               0.000                 1.000                 0.398 (+0.14)           0.000 (-0.02)
```
