# Fisher-retaining cascade protocol

No checkpoint was retrained. Frozen lambda and Q95 GCR decisions are unchanged. Beta was selected from {0, 0.25, 0.5, 0.75, 1} by mean inner-LOEO OSCR, then worst-fold OSCR, then smaller beta. Both Fisher and review accepted-branch scores were empirical-CDF calibrated only on Fisher-accepted known validation. No outer known or rogue sample was used for beta selection. Fixed operating-point metrics for the current and Fisher-retaining GCR must therefore be identical.
