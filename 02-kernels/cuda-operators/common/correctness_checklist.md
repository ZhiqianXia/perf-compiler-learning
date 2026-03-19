# Correctness Checklist

Use this checklist before calling an operator implementation done.

## Contract
- input tensor ranks are documented
- output tensor shape is documented
- supported dtypes are documented
- broadcasting rules are documented if relevant

## Edge Cases
- zero-sized dimensions
- non-multiple tile sizes
- negative values for exp/log/sqrt related ops
- very large magnitude values for softmax/sigmoid/tanh
- duplicate indices for gather/scatter style ops

## Reference
- CPU reference exists
- framework reference exists when practical
- tolerance is explicitly recorded

## Validation Cases
- tiny shape for manual inspection
- medium shape for random test
- adversarial shape for edge behavior
- shape that breaks naive tiling assumptions

## Output
- maximum absolute error
- maximum relative error when meaningful
- pass/fail decision
