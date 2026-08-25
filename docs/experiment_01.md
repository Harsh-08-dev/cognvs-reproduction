# EXP01 — Effect of Test-Time Fine-Tuning Budget

## Research Question

How does the amount of test-time fine-tuning affect
CogNVS output quality and computational cost?

## Motivation

CogNVS uses test-time fine-tuning as an important component
of its novel-view synthesis pipeline.

This experiment investigates how changing the number of
fine-tuning steps affects both reconstruction quality and
computational requirements.

## Hypothesis

Increasing the number of fine-tuning steps will generally
improve output quality, but the improvement will diminish
after a certain number of steps.

## Independent Variable

Number of test-time fine-tuning steps.

## Candidate Conditions

- 0 steps
- 50 steps
- 100 steps
- 200 steps

## Controlled Variables

- Input sequence
- Model checkpoint
- Camera trajectory
- Resolution
- Random seed
- Evaluation protocol

## Measurements

### Quality

- PSNR
- SSIM
- LPIPS

### Computational Cost

- Runtime
- Peak GPU memory

## Primary Comparison

Quality improvement versus additional computational cost.

## Expected Outputs

For every condition:

- Generated video
- Configuration file
- Runtime metadata
- Evaluation metrics

## Research Value

The experiment can reveal whether additional test-time
fine-tuning provides meaningful quality improvements relative
to its computational cost.

## Limitations

Results from a small number of sequences may not generalize
to the complete benchmark.

## Status

Planned