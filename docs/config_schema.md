# Experiment Configuration Schema

Every experiment run should be completely describable
through its configuration file.

## Required Fields

### experiment_id

Unique experiment identifier.

Example:

EXP01

### run_id

Unique identifier for a particular experimental condition.

Example:

EXP01_FT100

### input_sequence

Identifier of the input sequence.

Example:

seq01

### checkpoint

Model checkpoint used for the experiment.

Example:

baseline

### fine_tuning_steps

Number of test-time fine-tuning iterations.

Example:

100

### resolution

Resolution used for the experiment.

Example:

width: 576
height: 384

The exact resolution will be finalized after confirming
the implementation and dataset setup.

### seed

Random seed used for reproducibility.

Example:

42

### output_dir

Directory where outputs from this run are stored.

Example:

results/EXP01/FT100

### notes

Optional description of the run.