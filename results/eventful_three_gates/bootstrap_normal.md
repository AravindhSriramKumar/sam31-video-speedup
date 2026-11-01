# Paired bootstrap: difference from normal_final (70 common pairs, B = 200, seed 0)

Point estimate on all common pairs; [95% interval] of (run - reference) over bootstrap draws of pairs; p(run < ref) = share of draws where the run scores lower.

| run | cgF1 | pmF1 | IL_MCC | pHOTA | pDetA | pAssA | mask_mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal_final | 0.1561 | 0.3108 | 0.5024 | 0.5538 | 0.3739 | 0.8336 | 0.2993 |
| normal_three | 0.1430 (-0.0132 [-0.0302, +0.0001], p<ref 0.96) | 0.2969 (-0.0139 [-0.0321, +0.0086], p<ref 0.86) | 0.4815 (-0.0208 [-0.0715, +0.0000], p<ref 0.68) | 0.5353 (-0.0185 [-0.0365, -0.0023], p<ref 0.98) | 0.3556 (-0.0183 [-0.0412, +0.0023], p<ref 0.96) | 0.8169 (-0.0167 [-0.0327, +0.0044], p<ref 0.93) | 0.2801 (-0.0192 [-0.0423, +0.0065], p<ref 0.92) |
