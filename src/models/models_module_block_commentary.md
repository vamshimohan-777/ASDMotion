# src/models Module-By-Module Block Commentary

Auto-generated structural commentary for each code block in this module tree.
For each block, this document captures purpose, variable roles, called functions, and control flow.

## File: `src/models/image/__init__.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: none

- Top-level code blocks: none (file is currently empty or acts as placeholder).

## File: `src/models/nas_controller.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `copy`, `random`, `dataclasses: dataclass`, `numpy`, `torch`

### Block: `default_search_space`
- Location: line 9
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: none (direct arithmetic/control flow only)
- Flow summary: validates/transforms values and returns result

### Block: `_choice`
- Location: line 35
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `values`: local value used by this block
  - `rng`: local value used by this block
- Variables created in this block: `values`
- Functions/methods used: `list`, `ValueError`, `copy.deepcopy`, `rng.choice`
- Flow summary: `list` -> `ValueError` -> `copy.deepcopy` -> `rng.choice`

### Block: `sample_architecture`
- Location: line 42
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `search_space`: local value used by this block
  - `rng`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `_choice`, `items`
- Flow summary: `_choice` -> `items`

### Block: `compute_fitness`
- Location: line 50
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `metrics`: local value used by this block
- Variables created in this block: `auc`, `sens90`, `calibration`, `stability`, `eff_pen`, `eff_score`, `score`
- Functions/methods used: `float`, `metrics.get`, `np.clip`
- Flow summary: `float` -> `metrics.get` -> `np.clip`

### Class: `NASCandidate`
- Location: line 78
- Why this class exists: groups related state + methods into one reusable component.

- Methods: none

### Class: `MicroGeneticNAS`
- Location: line 84
- Why this class exists: groups related state + methods into one reusable component.

### Block: `MicroGeneticNAS.__init__`
- Location: line 85
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `population_size`: local value used by this block
  - `generations`: local value used by this block
  - `tournament_size`: local value used by this block
  - `mutation_rate`: local value used by this block
  - `crossover`: local value used by this block
  - `elite_count`: local value used by this block
  - `seed`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `int`, `max`, `float`, `np.clip`, `bool`, `random.Random`
- Flow summary: `int` -> `max` -> `float` -> `np.clip` -> `bool` -> `random.Random`

### Block: `MicroGeneticNAS._random_population`
- Location: line 103
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `search_space`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `NASCandidate`, `sample_architecture`, `range`
- Flow summary: `NASCandidate` -> `sample_architecture` -> `range`

### Block: `MicroGeneticNAS._evaluate`
- Location: line 109
- Why this block exists: Runs evaluation/inference logic and returns metrics/predictions.
- Variables (inputs) and roles:
  - `population`: local value used by this block
  - `evaluate_fn`: local value used by this block
- Variables created in this block: `cand`, `metrics`
- Functions/methods used: `evaluate_fn`, `copy.deepcopy`, `compute_fitness`, `population.sort`
- Flow summary: `evaluate_fn` -> `copy.deepcopy` -> `compute_fitness` -> `population.sort`

### Block: `MicroGeneticNAS._tournament_pick`
- Location: line 118
- Why this block exists: Converts/transforms structures into the format expected downstream.
- Variables (inputs) and roles:
  - `population`: local value used by this block
- Variables created in this block: `k`, `contenders`
- Functions/methods used: `min`, `len`, `self.rng.sample`, `contenders.sort`
- Flow summary: `min` -> `len` -> `self.rng.sample` -> `contenders.sort`

### Block: `MicroGeneticNAS._crossover`
- Location: line 124
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `arch_a`: local value used by this block
  - `arch_b`: local value used by this block
- Variables created in this block: `child`, `section`, `keys`, `k`
- Functions/methods used: `copy.deepcopy`, `child.keys`, `set`, `keys`, `arch_a.get`, `arch_b.get`, `self.rng.random`, `get`
- Flow summary: `copy.deepcopy` -> `child.keys` -> `set` -> `keys` -> `arch_a.get` -> `arch_b.get`

### Block: `MicroGeneticNAS._mutate`
- Location: line 137
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `arch`: local value used by this block
  - `search_space`: local value used by this block
- Variables created in this block: `out`, `key`, `fusion_dim`, `head_choices`
- Functions/methods used: `copy.deepcopy`, `out.items`, `list`, `params.keys`, `self.rng.random`, `_choice`, `int`, `get`
- Flow summary: `copy.deepcopy` -> `out.items` -> `list` -> `params.keys` -> `self.rng.random` -> `_choice`

### Block: `MicroGeneticNAS.evolve`
- Location: line 152
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `evaluate_fn`: local value used by this block
  - `search_space`: local value used by this block
  - `on_generation_end`: local value used by this block
- Variables created in this block: `search_space`, `population`, `history`, `gen`, `best`, `elites`, `next_pop`, `p1`, `p2`, `child_arch`
- Functions/methods used: `copy.deepcopy`, `default_search_space`, `self._random_population`, `range`, `self._evaluate`, `history.append`, `float`, `on_generation_end`, `len`, `self._tournament_pick`, `self._crossover`, `self._mutate`
- Flow summary: `copy.deepcopy` -> `default_search_space` -> `self._random_population` -> `range` -> `self._evaluate` -> `history.append`

### Class: `MicroNASController`
- Location: line 191
- Why this class exists: groups related state + methods into one reusable component.

### Block: `MicroNASController.__init__`
- Location: line 196
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `__init__`, `super`, `torch.nn.Parameter`, `torch.zeros`
- Flow summary: `__init__` -> `super` -> `torch.nn.Parameter` -> `torch.zeros`

### Block: `MicroNASController.arch_entropy_loss`
- Location: line 201
- Why this block exists: Computes a scalar optimization objective from predictions and targets.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `self._zero.sum`
- Flow summary: `self._zero.sum`

## File: `src/models/pipeline_model.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `copy`, `torch`, `torch.nn`, `src.models.video.mediapipe_layer.landmark_schema: DEFAULT_SCHEMA`, `src.models.video.motion.behavior_transformer: BehavioralTransformer`, `src.models.video.motion.event_encoder: ResNetMicroKineticEventEncoder`

### Class: `ASDPipeline`
- Location: line 11
- Why this class exists: groups related state + methods into one reusable component.

### Block: `ASDPipeline.__init__`
- Location: line 18
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `alpha`: local value used by this block
  - `K_max`: local value used by this block
  - `d_model`: local value used by this block
  - `dropout`: local value used by this block
  - `theta_high`: local value used by this block
  - `theta_low`: local value used by this block
  - `cnn_backbone`: local value used by this block
  - `nas_search_space`: local value used by this block
  - `num_event_types`: local value used by this block
  - `train_event_scorer_when_frozen`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `__init__`, `super`, `float`, `bool`, `int`, `max`, `self._build_from_architecture`
- Flow summary: `__init__` -> `super` -> `float` -> `bool` -> `int` -> `max`

### Block: `ASDPipeline._build_from_architecture`
- Location: line 63
- Why this block exists: Constructs reusable objects/components from configuration.
- Variables (inputs) and roles:
  - `arch`: local value used by this block
- Variables created in this block: `encoder_cfg`, `transformer_cfg`, `d_model`, `n_heads`, `candidates`
- Functions/methods used: `get`, `arch.get`, `int`, `max`, `ResNetMicroKineticEventEncoder`, `bool`, `float`, `transformer_cfg.get`, `encoder_cfg.get`, `BehavioralTransformer`
- Flow summary: `get` -> `arch.get` -> `int` -> `max` -> `ResNetMicroKineticEventEncoder` -> `bool`

### Block: `ASDPipeline._rebuild_with_arch`
- Location: line 93
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `arch`: local value used by this block
- Variables created in this block: `device`
- Functions/methods used: `next`, `self.parameters`, `torch.device`, `copy.deepcopy`, `self._build_from_architecture`, `self.to`
- Flow summary: `next` -> `self.parameters` -> `torch.device` -> `copy.deepcopy` -> `self._build_from_architecture` -> `self.to`

### Block: `ASDPipeline.freeze_motion_encoder`
- Location: line 102
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `train_event_scorer`: local value used by this block
- Variables created in this block: `train_event_scorer`, `p`
- Functions/methods used: `self.motion_encoder.parameters`, `bool`, `hasattr`, `self.motion_encoder.event_score_head.parameters`, `self.motion_encoder.eval`
- Flow summary: `self.motion_encoder.parameters` -> `bool` -> `hasattr` -> `self.motion_encoder.event_score_head.parameters` -> `self.motion_encoder.eval`

### Block: `ASDPipeline.unfreeze_upper_motion_layers`
- Location: line 114
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `num_blocks`: local value used by this block
- Variables created in this block: `num_blocks`, `p`, `blocks`, `block`, `name`, `branches`, `branch`, `block_list`
- Functions/methods used: `int`, `max`, `hasattr`, `self.motion_encoder.parameters`, `list`, `block.parameters`, `parameters`, `getattr`, `branch.parameters`, `branch.out_proj.parameters`
- Flow summary: `int` -> `max` -> `hasattr` -> `self.motion_encoder.parameters` -> `list` -> `block.parameters`

### Block: `ASDPipeline.freeze_cnns`
- Location: line 150
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `train_projection_heads`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `self.freeze_motion_encoder`
- Flow summary: `self.freeze_motion_encoder`

### Block: `ASDPipeline.train`
- Location: line 154
- Why this block exists: Runs training logic and returns optimization/training signals.
- Variables (inputs) and roles:
  - `mode`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `train`, `super`, `self.motion_encoder.eval`
- Flow summary: `train` -> `super` -> `self.motion_encoder.eval`

### Block: `ASDPipeline.trainable_parameters`
- Location: line 160
- Why this block exists: Runs training logic and returns optimization/training signals.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `self.parameters`
- Flow summary: `self.parameters`

### Block: `ASDPipeline.arch_parameters`
- Location: line 163
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: none (direct arithmetic/control flow only)
- Flow summary: validates/transforms values and returns result

### Block: `ASDPipeline.model_parameters`
- Location: line 166
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `self.trainable_parameters`
- Flow summary: `self.trainable_parameters`

### Block: `ASDPipeline.get_random_config`
- Location: line 170
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `nas_search_space`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: none (direct arithmetic/control flow only)
- Flow summary: validates/transforms values and returns result

### Block: `ASDPipeline.get_current_config`
- Location: line 174
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `copy.deepcopy`
- Flow summary: `copy.deepcopy`

### Block: `ASDPipeline.discretize_nas`
- Location: line 177
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `self.get_current_config`
- Flow summary: `self.get_current_config`

### Block: `ASDPipeline.apply_nas_architecture`
- Location: line 180
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `nas_arch`: local value used by this block
- Variables created in this block: `arch`, `enc`, `tr`, `wn`
- Functions/methods used: `copy.deepcopy`, `nas_arch.get`, `int`, `bool`, `float`, `str`, `self._rebuild_with_arch`
- Flow summary: `copy.deepcopy` -> `nas_arch.get` -> `int` -> `bool` -> `float` -> `str`

### Block: `ASDPipeline.forward`
- Location: line 220
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `inputs`: prepared model input dictionary/tensor
- Variables created in this block: `motion`, `B`, `S`, `W`, `J`, `F`, `joint_mask`, `window_valid`, `win_timestamps`, `flat_motion`, `flat_mask`, `flat_ts`, `enc_out`, `flat_window_emb`
- Functions/methods used: `motion.dim`, `ValueError`, `tuple`, `inputs.get`, `joint_mask.dim`, `sum`, `joint_mask.float`, `torch.ones`, `win_timestamps.dim`, `motion.reshape`, `joint_mask.reshape`, `win_timestamps.reshape`
- Flow summary: `motion.dim` -> `ValueError` -> `tuple` -> `inputs.get` -> `joint_mask.dim` -> `sum`

## File: `src/models/video/mediapipe_layer/landmark_schema.py`
- Module intent: Landmark schema for multimodal ASD motion learning.
- Key dependencies: `dataclasses: dataclass`

### Class: `LandmarkSchema`
- Location: line 26
- Why this class exists: groups related state + methods into one reusable component.

### Block: `LandmarkSchema.pose_slice`
- Location: line 32
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `slice`
- Flow summary: `slice`

### Block: `LandmarkSchema.left_hand_slice`
- Location: line 36
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: `start`
- Functions/methods used: `slice`
- Flow summary: `slice`

### Block: `LandmarkSchema.right_hand_slice`
- Location: line 41
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: `start`
- Functions/methods used: `slice`
- Flow summary: `slice`

### Block: `LandmarkSchema.face_slice`
- Location: line 46
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: `start`
- Functions/methods used: `slice`
- Flow summary: `slice`

### Block: `LandmarkSchema.total_joints`
- Location: line 51
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: none (direct arithmetic/control flow only)
- Flow summary: validates/transforms values and returns result

## File: `src/models/video/microkinetic_encoders/boundary_detector.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `torch`, `typing: List, Tuple`

### Class: `BoundaryDetector`
- Location: line 5
- Why this class exists: groups related state + methods into one reusable component.

### Block: `BoundaryDetector.__init__`
- Location: line 6
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `energy_threshold`: local value used by this block
  - `min_event_length`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: none (direct arithmetic/control flow only)
- Flow summary: validates/transforms values and returns result

### Block: `BoundaryDetector.detect`
- Location: line 10
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `energy`: local value used by this block
  - `mask`: validity mask tensor
- Variables created in this block: `t_len`, `segments`, `start`, `t`, `end`
- Functions/methods used: `range`, `segments.append`
- Flow summary: `range` -> `segments.append`

## File: `src/models/video/microkinetic_encoders/event_types.py`
- Module intent: Event taxonomy for the microkinetic encoder.
- Key dependencies: none

- Top-level code blocks: none (file is currently empty or acts as placeholder).

## File: `src/models/video/microkinetic_encoders/motion_ssl_encoder.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `torch`, `torch.nn`, `src.models.video.mediapipe_layer.landmark_schema: DEFAULT_SCHEMA`

### Class: `TemporalConvResidualBlock`
- Location: line 7
- Why this class exists: groups related state + methods into one reusable component.

### Block: `TemporalConvResidualBlock.__init__`
- Location: line 8
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `channels`: local value used by this block
  - `kernel_size`: local value used by this block
  - `dilation`: local value used by this block
  - `dropout`: local value used by this block
- Variables created in this block: `padding`
- Functions/methods used: `__init__`, `super`, `int`, `nn.Conv1d`, `nn.BatchNorm1d`, `nn.GELU`, `nn.Dropout`, `float`
- Flow summary: `__init__` -> `super` -> `int` -> `nn.Conv1d` -> `nn.BatchNorm1d` -> `nn.GELU`

### Block: `TemporalConvResidualBlock.forward`
- Location: line 32
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `h`
- Functions/methods used: `self.conv1`, `self.bn1`, `self.act`, `self.conv2`, `self.bn2`, `self.drop`
- Flow summary: `self.conv1` -> `self.bn1` -> `self.act` -> `self.conv2` -> `self.bn2` -> `self.drop`

### Class: `JointWiseTemporalBranch`
- Location: line 42
- Why this class exists: groups related state + methods into one reusable component.

### Block: `JointWiseTemporalBranch.__init__`
- Location: line 43
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `in_features`: local value used by this block
  - `hidden_dim`: local value used by this block
  - `out_dim`: module/model output dictionary
  - `kernel_sizes`: local value used by this block
  - `use_dilation`: local value used by this block
  - `dropout`: local value used by this block
  - `joint_pool`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `__init__`, `super`, `strip`, `lower`, `str`, `ValueError`, `nn.Sequential`, `nn.Conv1d`, `int`, `nn.BatchNorm1d`, `nn.GELU`, `nn.ModuleList`
- Flow summary: `__init__` -> `super` -> `strip` -> `lower` -> `str` -> `ValueError`

### Block: `JointWiseTemporalBranch._masked_mean`
- Location: line 87
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
  - `valid`: local value used by this block
- Variables created in this block: `mask`, `denom`
- Functions/methods used: `unsqueeze`, `valid.float`, `clamp`, `mask.sum`, `sum`
- Flow summary: `unsqueeze` -> `valid.float` -> `clamp` -> `mask.sum` -> `sum`

### Block: `JointWiseTemporalBranch._joint_pooling`
- Location: line 92
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `joint_features`: local value used by this block
  - `joint_valid`: local value used by this block
- Variables created in this block: `scores`, `weights`, `pooled`
- Functions/methods used: `self._masked_mean`, `squeeze`, `self.joint_attention`, `scores.masked_fill`, `torch.softmax`, `sum`, `weights.unsqueeze`
- Flow summary: `self._masked_mean` -> `squeeze` -> `self.joint_attention` -> `scores.masked_fill` -> `torch.softmax` -> `sum`

### Block: `JointWiseTemporalBranch.forward`
- Location: line 102
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `x`: input tensor/features
  - `joint_valid`: local value used by this block
- Variables created in this block: `b`, `t`, `j`, `f`, `h`, `block`, `joint_valid`, `pooled`
- Functions/methods used: `contiguous`, `reshape`, `x.permute`, `self.input_proj`, `block`, `squeeze`, `self.time_pool`, `h.reshape`, `sum`, `x.abs`, `self._joint_pooling`, `self.readout`
- Flow summary: `contiguous` -> `reshape` -> `x.permute` -> `self.input_proj` -> `block` -> `squeeze`

### Class: `MultiBranchMotionEncoderSSL`
- Location: line 118
- Why this class exists: groups related state + methods into one reusable component.

### Block: `MultiBranchMotionEncoderSSL.__init__`
- Location: line 119
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `schema`: local value used by this block
  - `in_features`: local value used by this block
  - `branch_hidden_dim`: local value used by this block
  - `branch_out_dim`: local value used by this block
  - `embedding_dim`: local value used by this block
  - `kernel_sizes`: local value used by this block
  - `use_dilation`: local value used by this block
  - `dropout`: local value used by this block
  - `joint_pool`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `__init__`, `super`, `int`, `JointWiseTemporalBranch`, `nn.Sequential`, `nn.Linear`, `nn.LayerNorm`
- Flow summary: `__init__` -> `super` -> `int` -> `JointWiseTemporalBranch` -> `nn.Sequential` -> `nn.Linear`

### Block: `MultiBranchMotionEncoderSSL.forward`
- Location: line 167
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `pose_x`, `hand_x`, `face_x`, `pose_valid`, `hand_valid`, `face_valid`, `pose_z`, `hand_z`, `face_z`, `fused`
- Functions/methods used: `sum`, `pose_x.abs`, `hand_x.abs`, `face_x.abs`, `self.pose_branch`, `self.hand_branch`, `self.face_branch`, `torch.cat`, `self.fusion`
- Flow summary: `sum` -> `pose_x.abs` -> `hand_x.abs` -> `face_x.abs` -> `self.pose_branch` -> `self.hand_branch`

### Block: `freeze_encoder`
- Location: line 189
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `module`: local value used by this block
- Variables created in this block: `p`
- Functions/methods used: `module.parameters`, `module.eval`
- Flow summary: `module.parameters` -> `module.eval`

## File: `src/models/video/motion/__init__.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `src.models.video.motion.behavior_transformer: BehavioralTransformer`, `src.models.video.motion.encoder: MultiBranchMotionEncoder`, `src.models.video.motion.event_encoder: ResNetMicroKineticEventEncoder`

- Top-level code blocks: none (file is currently empty or acts as placeholder).

## File: `src/models/video/motion/behavior_transformer.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `math`, `torch`, `torch.nn`

### Class: `PositionalEncoding`
- Location: line 7
- Why this class exists: groups related state + methods into one reusable component.

### Block: `PositionalEncoding.__init__`
- Location: line 8
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `d_model`: local value used by this block
  - `max_len`: local value used by this block
- Variables created in this block: `pe`, `pos`, `div_term`
- Functions/methods used: `__init__`, `super`, `torch.zeros`, `unsqueeze`, `torch.arange`, `torch.exp`, `math.log`, `torch.sin`, `torch.cos`, `self.register_buffer`, `pe.unsqueeze`
- Flow summary: `__init__` -> `super` -> `torch.zeros` -> `unsqueeze` -> `torch.arange` -> `torch.exp`

### Block: `PositionalEncoding.forward`
- Location: line 19
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `x.size`
- Flow summary: `x.size`

### Class: `BehavioralTransformer`
- Location: line 24
- Why this class exists: groups related state + methods into one reusable component.

### Block: `BehavioralTransformer.__init__`
- Location: line 25
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `d_model`: local value used by this block
  - `n_heads`: local value used by this block
  - `n_layers`: local value used by this block
  - `dim_ff`: local value used by this block
  - `dropout`: local value used by this block
- Variables created in this block: `enc_layer`
- Functions/methods used: `__init__`, `super`, `PositionalEncoding`, `nn.Sequential`, `nn.Linear`, `nn.GELU`, `nn.TransformerEncoderLayer`, `nn.TransformerEncoder`, `nn.LayerNorm`, `nn.Dropout`, `nn.Tanh`
- Flow summary: `__init__` -> `super` -> `PositionalEncoding` -> `nn.Sequential` -> `nn.Linear` -> `nn.GELU`

### Block: `BehavioralTransformer.forward`
- Location: line 63
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `window_embeddings`: local value used by this block
  - `window_mask`: local value used by this block
  - `event_times`: local value used by this block
  - `aggregation`: local value used by this block
- Variables created in this block: `h`, `t`, `key_padding_mask`, `window_mask`, `valid`, `scores`, `k`, `topk_vals`, `_`, `pooled`, `logit`, `a`, `w`
- Functions/methods used: `self.pos`, `unsqueeze`, `torch.log1p`, `event_times.float`, `self.time_mlp`, `window_mask.bool`, `self.encoder`, `self.norm`, `torch.ones`, `h.size`, `window_mask.float`, `squeeze`
- Flow summary: `self.pos` -> `unsqueeze` -> `torch.log1p` -> `event_times.float` -> `self.time_mlp` -> `window_mask.bool`

## File: `src/models/video/motion/blocks.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `torch`, `torch.nn`

### Class: `MicroKineticBlock`
- Location: line 5
- Why this class exists: groups related state + methods into one reusable component.

### Block: `MicroKineticBlock.__init__`
- Location: line 6
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `channels`: local value used by this block
  - `kernel_size`: local value used by this block
  - `dilation`: local value used by this block
  - `dropout`: local value used by this block
  - `residual`: local value used by this block
- Variables created in this block: `pad`
- Functions/methods used: `__init__`, `super`, `nn.Conv1d`, `nn.BatchNorm1d`, `nn.GELU`, `nn.Dropout`, `bool`
- Flow summary: `__init__` -> `super` -> `nn.Conv1d` -> `nn.BatchNorm1d` -> `nn.GELU` -> `nn.Dropout`

### Block: `MicroKineticBlock.forward`
- Location: line 31
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `h`
- Functions/methods used: `self.depthwise`, `self.pointwise`, `self.norm`, `self.act`, `self.drop`
- Flow summary: `self.depthwise` -> `self.pointwise` -> `self.norm` -> `self.act` -> `self.drop`

## File: `src/models/video/motion/encoder.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `random`, `torch`, `torch.nn`, `src.models.video.mediapipe_layer.landmark_schema: DEFAULT_SCHEMA`, `src.models.video.motion.blocks: MicroKineticBlock`

### Class: `TemporalBranchEncoder`
- Location: line 10
- Why this class exists: groups related state + methods into one reusable component.

### Block: `TemporalBranchEncoder.__init__`
- Location: line 11
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `n_joints`: local value used by this block
  - `in_feat`: local value used by this block
  - `channels`: local value used by this block
  - `n_blocks`: local value used by this block
  - `kernel_size`: local value used by this block
  - `use_dilation`: local value used by this block
  - `residual`: local value used by this block
  - `dropout`: local value used by this block
  - `out_dim`: module/model output dictionary
- Variables created in this block: `in_ch`, `blocks`, `i`, `dilation`
- Functions/methods used: `__init__`, `super`, `int`, `nn.Sequential`, `nn.Conv1d`, `nn.BatchNorm1d`, `nn.GELU`, `range`, `blocks.append`, `MicroKineticBlock`, `float`, `bool`
- Flow summary: `__init__` -> `super` -> `int` -> `nn.Sequential` -> `nn.Conv1d` -> `nn.BatchNorm1d`

### Block: `TemporalBranchEncoder.forward`
- Location: line 52
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `B`, `W`, `J`, `F`, `h`
- Functions/methods used: `transpose`, `x.reshape`, `self.input_proj`, `self.blocks`, `self.out_proj`
- Flow summary: `transpose` -> `x.reshape` -> `self.input_proj` -> `self.blocks` -> `self.out_proj`

### Class: `MultiBranchMotionEncoder`
- Location: line 61
- Why this class exists: groups related state + methods into one reusable component.

### Block: `MultiBranchMotionEncoder.__init__`
- Location: line 62
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `schema`: local value used by this block
  - `in_feat`: local value used by this block
  - `branch_channels`: local value used by this block
  - `branch_blocks`: local value used by this block
  - `kernel_size`: local value used by this block
  - `use_dilation`: local value used by this block
  - `residual`: local value used by this block
  - `branch_dropout`: local value used by this block
  - `embedding_dim`: local value used by this block
  - `fusion_dim`: local value used by this block
  - `modality_dropout`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `__init__`, `super`, `int`, `float`, `TemporalBranchEncoder`, `nn.Sequential`, `nn.Linear`, `nn.GELU`, `nn.LayerNorm`
- Flow summary: `__init__` -> `super` -> `int` -> `float` -> `TemporalBranchEncoder` -> `nn.Sequential`

### Block: `MultiBranchMotionEncoder._apply_modality_dropout`
- Location: line 121
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `pose_z`: local value used by this block
  - `hand_z`: local value used by this block
  - `face_z`: local value used by this block
- Variables created in this block: `p`, `mode`, `hand_z`, `face_z`
- Functions/methods used: `random.random`, `random.choice`, `torch.zeros_like`
- Flow summary: `random.random` -> `random.choice` -> `torch.zeros_like`

### Block: `MultiBranchMotionEncoder.forward`
- Location: line 138
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `motion_windows`: local value used by this block
  - `joint_mask`: local value used by this block
- Variables created in this block: `pose_x`, `hand_x`, `face_x`, `pose_z`, `hand_z`, `face_z`, `fused`
- Functions/methods used: `self.pose_encoder`, `self.hand_encoder`, `self.face_encoder`, `self._apply_modality_dropout`, `torch.cat`, `self.fuse`
- Flow summary: `self.pose_encoder` -> `self.hand_encoder` -> `self.face_encoder` -> `self._apply_modality_dropout` -> `torch.cat` -> `self.fuse`

## File: `src/models/video/motion/event_encoder.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `torch`, `torch.nn`, `torch.nn.functional`, `torchvision: models`, `src.models.video.motion.blocks: MicroKineticBlock`

### Class: `ResNetMicroKineticEventEncoder`
- Location: line 9
- Why this class exists: groups related state + methods into one reusable component.

### Block: `ResNetMicroKineticEventEncoder.__init__`
- Location: line 26
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `d_model`: local value used by this block
  - `temporal_channels`: local value used by this block
  - `micro_blocks`: local value used by this block
  - `kernel_size`: local value used by this block
  - `use_dilation`: local value used by this block
  - `residual`: local value used by this block
  - `dropout`: local value used by this block
  - `k_max`: local value used by this block
- Variables created in this block: `backbone`, `blocks`, `i`, `dilation`
- Functions/methods used: `__init__`, `super`, `int`, `max`, `models.resnet18`, `nn.Sequential`, `list`, `backbone.children`, `nn.Linear`, `nn.GELU`, `nn.LayerNorm`, `nn.Conv1d`
- Flow summary: `__init__` -> `super` -> `int` -> `max` -> `models.resnet18` -> `nn.Sequential`

### Block: `ResNetMicroKineticEventEncoder._motion_to_rgb_image`
- Location: line 81
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `motion_windows`: local value used by this block
- Variables created in this block: `b`, `w`, `j`, `f`, `x`
- Functions/methods used: `ValueError`, `contiguous`, `permute`, `motion_windows.reshape`, `F.interpolate`
- Flow summary: `ValueError` -> `contiguous` -> `permute` -> `motion_windows.reshape` -> `F.interpolate`

### Block: `ResNetMicroKineticEventEncoder._frame_valid_mask`
- Location: line 92
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `joint_mask`: local value used by this block
  - `b`: local value used by this block
  - `w`: local value used by this block
  - `device`: runtime compute device (cpu/cuda)
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `torch.ones`, `sum`, `joint_mask.float`
- Flow summary: `torch.ones` -> `sum` -> `joint_mask.float`

### Block: `ResNetMicroKineticEventEncoder.forward`
- Location: line 98
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `motion_windows`: local value used by this block
  - `joint_mask`: local value used by this block
  - `timestamps`: time index tensor
  - `return_events`: local value used by this block
- Variables created in this block: `b`, `w`, `_`, `device`, `img`, `feat_2d`, `frame_feat`, `x`, `frame_valid`, `frame_logits`, `masked_frame_logits`, `frame_scores`, `k`, `topk_logits`
- Functions/methods used: `self._motion_to_rgb_image`, `flatten`, `self.resnet_backbone`, `reshape`, `self.frame_proj`, `frame_feat.transpose`, `self.temporal_in`, `self.temporal_blocks`, `self.temporal_out`, `x.transpose`, `self._frame_valid_mask`, `squeeze`
- Flow summary: `self._motion_to_rgb_image` -> `flatten` -> `self.resnet_backbone` -> `reshape` -> `self.frame_proj` -> `frame_feat.transpose`

## File: `src/models/video/transformer_reasoning/__init__.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: none

- Top-level code blocks: none (file is currently empty or acts as placeholder).

## File: `src/models/video/transformer_reasoning/event_transformer.py`
- Module intent: Temporal Transformer with sinusoidal positional encoding and learned time-gap embedding.
- Key dependencies: `torch`, `torch.nn`, `contextlib: contextmanager`, `src.models.video.microkinetic_encoders.event_types: NUM_EVENT_TYPES`

### Class: `SinusoidalTimeEmbedding`
- Location: line 11
- Why this class exists: groups related state + methods into one reusable component.

### Block: `SinusoidalTimeEmbedding.__init__`
- Location: line 12
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `d_model`: local value used by this block
- Variables created in this block: `inv_freq`
- Functions/methods used: `__init__`, `super`, `float`, `torch.arange`, `self.register_buffer`
- Flow summary: `__init__` -> `super` -> `float` -> `torch.arange` -> `self.register_buffer`

### Block: `SinusoidalTimeEmbedding.forward`
- Location: line 18
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `time_positions`: local value used by this block
- Variables created in this block: `t`, `freqs`, `emb`
- Functions/methods used: `float`, `time_positions.unsqueeze`, `unsqueeze`, `self.inv_freq.unsqueeze`, `torch.cat`, `freqs.sin`, `freqs.cos`
- Flow summary: `float` -> `time_positions.unsqueeze` -> `unsqueeze` -> `self.inv_freq.unsqueeze` -> `torch.cat` -> `freqs.sin`

### Class: `TimeGapEmbedding`
- Location: line 25
- Why this class exists: groups related state + methods into one reusable component.

### Block: `TimeGapEmbedding.__init__`
- Location: line 26
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `d_model`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `__init__`, `super`, `nn.Sequential`, `nn.Linear`, `nn.GELU`
- Flow summary: `__init__` -> `super` -> `nn.Sequential` -> `nn.Linear` -> `nn.GELU`

### Block: `TimeGapEmbedding.forward`
- Location: line 34
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `delta_t`: local value used by this block
- Variables created in this block: `dt`
- Functions/methods used: `unsqueeze`, `torch.log1p`, `delta_t.float`, `self.mlp`
- Flow summary: `unsqueeze` -> `torch.log1p` -> `delta_t.float` -> `self.mlp`

### Class: `HookableTransformerEncoderLayer`
- Location: line 40
- Why this class exists: groups related state + methods into one reusable component.

### Block: `HookableTransformerEncoderLayer.forward`
- Location: line 41
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `src`: local value used by this block
  - `src_mask`: local value used by this block
  - `src_key_padding_mask`: local value used by this block
  - `is_causal`: local value used by this block
- Variables created in this block: `x`
- Functions/methods used: `self._sa_block`, `self.norm1`, `self._ff_block`, `self.norm2`
- Flow summary: `self._sa_block` -> `self.norm1` -> `self._ff_block` -> `self.norm2`

### Block: `HookableTransformerEncoderLayer._sa_block`
- Location: line 51
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
  - `attn_mask`: local value used by this block
  - `key_padding_mask`: local value used by this block
  - `is_causal`: local value used by this block
- Variables created in this block: `capture_list`, `need_weights`, `out`, `x`, `weights`
- Functions/methods used: `getattr`, `self.self_attn`, `capture_list.append`, `weights.detach`, `self.dropout1`
- Flow summary: `getattr` -> `self.self_attn` -> `capture_list.append` -> `weights.detach` -> `self.dropout1`

### Block: `HookableTransformerEncoderLayer._ff_block`
- Location: line 72
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `x`
- Functions/methods used: `self.linear2`, `self.dropout`, `self.activation`, `self.linear1`, `self.dropout2`
- Flow summary: `self.linear2` -> `self.dropout` -> `self.activation` -> `self.linear1` -> `self.dropout2`

### Class: `TemporalTransformer`
- Location: line 77
- Why this class exists: groups related state + methods into one reusable component.

### Block: `TemporalTransformer.__init__`
- Location: line 78
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `d_model`: local value used by this block
  - `n_heads`: local value used by this block
  - `scalars_dim`: local value used by this block
  - `num_encoder_layers`: local value used by this block
  - `dim_ff`: local value used by this block
  - `dropout`: local value used by this block
  - `activation`: local value used by this block
  - `num_event_types`: local value used by this block
  - `event_type_emb_dim`: local value used by this block
- Variables created in this block: `fuse_in`, `encoder_layer`
- Functions/methods used: `__init__`, `super`, `SinusoidalTimeEmbedding`, `TimeGapEmbedding`, `nn.Embedding`, `nn.Sequential`, `nn.Linear`, `nn.GELU`, `nn.LayerNorm`, `HookableTransformerEncoderLayer`, `nn.TransformerEncoder`, `nn.Dropout`
- Flow summary: `__init__` -> `super` -> `SinusoidalTimeEmbedding` -> `TimeGapEmbedding` -> `nn.Embedding` -> `nn.Sequential`

### Block: `TemporalTransformer._init_weights`
- Location: line 139
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: `module`, `m`
- Functions/methods used: `module.modules`, `isinstance`, `nn.init.xavier_uniform_`, `nn.init.zeros_`, `torch.no_grad`, `bias.fill_`
- Flow summary: `module.modules` -> `isinstance` -> `nn.init.xavier_uniform_` -> `nn.init.zeros_` -> `torch.no_grad` -> `bias.fill_`

### Block: `TemporalTransformer.capture_attention`
- Location: line 150
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: `layer`
- Functions/methods used: `hasattr`
- Flow summary: `hasattr`

### Block: `TemporalTransformer.forward`
- Location: line 163
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `tokens`, `attn_mask`, `time_positions`, `event_type_id`, `token_conf`, `event_scalars`, `delta_t`, `B`, `K`, `D`, `conf`, `type_emb`, `fused`, `src_key_padding_mask`
- Functions/methods used: `x.get`, `torch.zeros`, `token_conf.unsqueeze`, `self.type_emb`, `torch.cat`, `self.token_fuse`, `self.time_emb`, `self.time_gap_emb`, `self.encoder`, `self.norm`, `unsqueeze`, `attn_mask.float`
- Flow summary: `x.get` -> `torch.zeros` -> `token_conf.unsqueeze` -> `self.type_emb` -> `torch.cat` -> `self.token_fuse`
