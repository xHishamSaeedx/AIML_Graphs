# Calibration Plot Visualizations

This module demonstrates six reliability-diagram scenarios that frequently
surface when evaluating safety-critical ML systems (LLM safety, autonomous
perception, medical risk scoring, finance, etc.). Run the script via:

```
python calibration_plot_visualizations.py
```

Each subplot uses synthetic probabilities/labels to highlight specific
calibration behaviors. Think of each scenario like a data-science “spot the
pattern” exercise:

1. **Perfect calibration**

   - _What you see:_ The curve hugs the diagonal, meaning predicted probabilities
     (x-axis) match observed outcomes (y-axis).
   - _Simple example:_ If a medical triage model says 40% risk for 100 patients,
     about 40 actually need urgent care. The diagonal tells you the model already
     speaks “truthful probabilities.”

2. **Overconfident model**

   - _What you see:_ Curve sits below the diagonal. The model claims higher odds
     than reality delivers.
   - _Simple example:_ A fraud detector flags purchases with a “90% chance” of
     being fraud, yet only 50% truly are. Operations teams overreact because the
     model sounds too certain.

3. **Underconfident model**

   - _What you see:_ Curve above the diagonal. Predictions are conservative and
     actual event rates are higher.
   - _Simple example:_ An insurance risk score outputs 20% probability for roof
     leaks, but 40% actually leak. Claims adjusters under-prepare because the
     model whispers instead of speaking up.

4. **High-variance / noisy calibration**

   - _What you see:_ Wiggly line with jumps. Usually happens when you have few
     samples per bin or lots of randomness.
   - _Simple example:_ A robotics perception team tests a tiny dataset (maybe 50
     rare pedestrians). Some bins show 0%, some 100%, not because the model is
     wildly wrong but because the sample is small. The variance warns you to
     gather more data before trusting the shape.

5. **Overfitting (train vs validation)**

   - _What you see:_ Training curve stays near the diagonal, but validation
     curves drift—often below it.
   - _Simple example:_ On past claims, a legal risk model looks perfect. When
     deployed in a new region, predicted 70% probabilities turn into 30% real
     outcomes. The gap signals the model memorized quirks of the training set.

6. **Severe class imbalance**
   - _What you see:_ Most bins crowd near the origin; the curve may spike later.
     Calibration exposes how rarely positives occur.
   - _Simple example:_ In a hospital readmission model, 95% of patients never
     return. Even when the model predicts 0.2 probability, the actual frequency
     might be 0.05 because there just aren’t many positives. Calibration helps
     you notice that “low probabilities everywhere” is a data issue, not
     necessarily a model flaw.

The script relies only on `numpy`, `matplotlib`, `seaborn`, and
`sklearn.calibration.calibration_curve`, making it easy to adapt for internal
education or reliability reviews. Adjust the RNG seed, binning strategy, or
annotations to tailor the visuals for your team.
