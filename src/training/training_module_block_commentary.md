# src/training Module-By-Module Block Commentary

Auto-generated structural commentary for each code block in this module tree.
For each block, this document captures purpose, variable roles, called functions, and control flow.

## File: `src/training/checkpoints.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `json`, `os`, `tempfile`, `torch`

### Class: `CheckpointManager`
- Location: line 8
- Why this class exists: groups related state + methods into one reusable component.

### Block: `CheckpointManager.__init__`
- Location: line 9
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `root_dir`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `str`, `os.makedirs`
- Flow summary: `str` -> `os.makedirs`

### Block: `CheckpointManager._atomic_json_dump`
- Location: line 13
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `path`: filesystem path string
  - `payload`: local value used by this block
- Variables created in this block: `fd`, `tmp_path`
- Functions/methods used: `os.makedirs`, `os.path.dirname`, `tempfile.mkstemp`, `os.close`, `open`, `json.dump`, `os.replace`, `os.path.exists`, `os.remove`
- Flow summary: `os.makedirs` -> `os.path.dirname` -> `tempfile.mkstemp` -> `os.close` -> `open` -> `json.dump`

### Block: `CheckpointManager.save_model`
- Location: line 25
- Why this block exists: Persists artifacts for reproducibility or downstream stages.
- Variables (inputs) and roles:
  - `filename`: local value used by this block
  - `payload`: local value used by this block
- Variables created in this block: `path`
- Functions/methods used: `os.path.join`, `torch.save`
- Flow summary: `os.path.join` -> `torch.save`

### Block: `CheckpointManager.save_json`
- Location: line 30
- Why this block exists: Persists artifacts for reproducibility or downstream stages.
- Variables (inputs) and roles:
  - `filename`: local value used by this block
  - `payload`: local value used by this block
- Variables created in this block: `path`
- Functions/methods used: `os.path.join`, `self._atomic_json_dump`
- Flow summary: `os.path.join` -> `self._atomic_json_dump`

## File: `src/training/dataset.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `csv`, `json`, `os`, `random`, `urllib.parse`, `urllib.request`, `numpy`, `torch`, `torch.utils.data: Dataset`, `src.models.video.mediapipe_layer.landmark_schema: DEFAULT_SCHEMA`, `src.pipeline.preprocess: VideoProcessor`, `src.utils.video_id: make_video_id`

### Block: `_safe_text`
- Location: line 17
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `value`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `strip`, `str`
- Flow summary: `strip` -> `str`

### Block: `_moving_average_1d`
- Location: line 23
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
  - `k`: local value used by this block
- Variables created in this block: `k`, `pad`, `kernel`, `xp`
- Functions/methods used: `int`, `max`, `np.ones`, `kernel.sum`, `np.pad`, `np.convolve`
- Flow summary: `int` -> `max` -> `np.ones` -> `kernel.sum` -> `np.pad` -> `np.convolve`

### Block: `_is_http_link`
- Location: line 36
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `text`: local value used by this block
- Variables created in this block: `t`
- Functions/methods used: `lower`, `_safe_text`, `t.startswith`
- Flow summary: `lower` -> `_safe_text` -> `t.startswith`

### Block: `_fill_missing_1d`
- Location: line 41
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `values`: local value used by this block
  - `valid`: local value used by this block
- Variables created in this block: `out`, `idx`, `first`, `last`, `miss`, `m`, `left`, `right`, `l`, `r`, `alpha`
- Functions/methods used: `values.copy`, `valid.sum`, `np.where`, `max`
- Flow summary: `values.copy` -> `valid.sum` -> `np.where` -> `max`

### Class: `VideoDataset`
- Location: line 69
- Why this class exists: groups related state + methods into one reusable component.

### Block: `VideoDataset.__init__`
- Location: line 78
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `csv_path`: local value used by this block
  - `sequence_length`: local value used by this block
  - `is_training`: local value used by this block
  - `require_label`: local value used by this block
  - `use_preprocessed`: local value used by this block
  - `processed_root`: local value used by this block
  - `window_sizes`: local value used by this block
  - `windows_per_video`: local value used by this block
  - `eval_windows_per_video`: local value used by this block
  - `frame_stride`: local value used by this block
  - `max_frames`: local value used by this block
  - `cache_enabled`: local value used by this block
  - `smooth_kernel`: local value used by this block
- Variables created in this block: `reader`, `fieldnames`, `action_type`, `skeleton_source`, `label`, `video_path`, `subject_id`, `entry`, `action`
- Functions/methods used: `int`, `bool`, `str`, `tuple`, `sorted`, `max`, `VideoProcessor`, `os.path.join`, `open`, `csv.DictReader`, `ValueError`, `enumerate`
- Flow summary: `int` -> `bool` -> `str` -> `tuple` -> `sorted` -> `max`

### Block: `VideoDataset.__len__`
- Location: line 163
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `len`
- Flow summary: `len`

### Block: `VideoDataset._candidate_ids`
- Location: line 166
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `entry`: local value used by this block
- Variables created in this block: `video_path`, `subject_id`, `label`, `ids`, `prev`, `legacy`
- Functions/methods used: `entry.get`, `ids.append`, `make_video_id`
- Flow summary: `entry.get` -> `ids.append` -> `make_video_id`

### Block: `VideoDataset._parse_action_type`
- Location: line 181
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `row`: local value used by this block
- Variables created in this block: `key`, `value`
- Functions/methods used: `_safe_text`, `row.get`
- Flow summary: `_safe_text` -> `row.get`

### Block: `VideoDataset._parse_label`
- Location: line 188
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `row`: local value used by this block
- Variables created in this block: `text`
- Functions/methods used: `_safe_text`, `row.get`, `ValueError`, `float`
- Flow summary: `_safe_text` -> `row.get` -> `ValueError` -> `float`

### Block: `VideoDataset._parse_skeleton_source`
- Location: line 202
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `row`: local value used by this block
- Variables created in this block: `keys`, `key`, `value`
- Functions/methods used: `_safe_text`, `row.get`
- Flow summary: `_safe_text` -> `row.get`

### Block: `VideoDataset._parse_ntu_skeleton_file`
- Location: line 225
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `path`: filesystem path string
- Variables created in this block: `lines`, `p`, `n_frames`, `frames`, `max_joints`, `_`, `n_bodies`, `best`, `best_valid`, `n_joints`, `joints`, `vals`, `x`, `y`
- Functions/methods used: `open`, `ln.strip`, `ValueError`, `int`, `float`, `range`, `max`, `len`, `split`, `joints.append`, `np.asarray`, `sum`
- Flow summary: `open` -> `ln.strip` -> `ValueError` -> `int` -> `float` -> `range`

### Block: `VideoDataset._download_external_file`
- Location: line 312
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `url`: local value used by this block
- Variables created in this block: `parsed`, `name`, `safe_name`, `local_path`, `data`
- Functions/methods used: `os.makedirs`, `urllib.parse.urlparse`, `os.path.basename`, `replace`, `name.replace`, `os.path.join`, `os.path.exists`, `os.path.getsize`, `urllib.request.urlopen`, `resp.read`, `open`, `f.write`
- Flow summary: `os.makedirs` -> `urllib.parse.urlparse` -> `os.path.basename` -> `replace` -> `name.replace` -> `os.path.join`

### Block: `VideoDataset._normalize_skeleton_arrays`
- Location: line 326
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `landmarks`: local value used by this block
  - `mask`: validity mask tensor
  - `timestamps`: time index tensor
- Variables created in this block: `arr`, `t`, `d`, `j`, `c`, `z`, `target_j`, `pad_j`, `mask`, `timestamps`
- Functions/methods used: `np.asarray`, `ValueError`, `arr.reshape`, `np.zeros`, `np.concatenate`, `int`, `np.pad`, `astype`, `sum`, `np.abs`, `np.arange`, `reshape`
- Flow summary: `np.asarray` -> `ValueError` -> `arr.reshape` -> `np.zeros` -> `np.concatenate` -> `int`

### Block: `VideoDataset._load_skeleton_file`
- Location: line 383
- Why this block exists: Loads configuration/state/data required by later runtime blocks.
- Variables (inputs) and roles:
  - `path`: filesystem path string
- Variables created in this block: `ext`, `quality`, `data`, `keys`, `lmk_key`, `landmarks`, `mask`, `timestamps`, `arr`, `m`, `ts`, `obj`, `arr3`
- Functions/methods used: `lower`, `os.path.splitext`, `np.load`, `set`, `ValueError`, `self._normalize_skeleton_arrays`, `isinstance`, `arr.item`, `obj.get`, `open`, `json.load`, `self._parse_ntu_skeleton_file`
- Flow summary: `lower` -> `os.path.splitext` -> `np.load` -> `set` -> `ValueError` -> `self._normalize_skeleton_arrays`

### Block: `VideoDataset._load_from_skeleton_source`
- Location: line 433
- Why this block exists: Loads configuration/state/data required by later runtime blocks.
- Variables (inputs) and roles:
  - `entry`: local value used by this block
- Variables created in this block: `source`, `path`
- Functions/methods used: `_safe_text`, `entry.get`, `_is_http_link`, `self._download_external_file`, `os.path.isabs`, `os.path.abspath`, `os.path.exists`, `FileNotFoundError`, `self._load_skeleton_file`
- Flow summary: `_safe_text` -> `entry.get` -> `_is_http_link` -> `self._download_external_file` -> `os.path.isabs` -> `os.path.abspath`

### Block: `VideoDataset._load_preprocessed`
- Location: line 447
- Why this block exists: Loads configuration/state/data required by later runtime blocks.
- Variables (inputs) and roles:
  - `entry`: local value used by this block
- Variables created in this block: `base_dir`, `vid`, `candidate`, `landmarks_path`, `mask_path`, `timestamps_path`, `quality_path`, `landmarks`, `mask`, `timestamps`, `quality`
- Functions/methods used: `self._candidate_ids`, `os.path.join`, `os.path.exists`, `astype`, `np.load`, `open`, `json.load`
- Flow summary: `self._candidate_ids` -> `os.path.join` -> `os.path.exists` -> `astype` -> `np.load` -> `open`

### Block: `VideoDataset._load_from_video`
- Location: line 479
- Why this block exists: Loads configuration/state/data required by later runtime blocks.
- Variables (inputs) and roles:
  - `entry`: local value used by this block
- Variables created in this block: `result`, `frames`, `T`, `J`, `landmarks`, `mask`, `timestamps`, `quality`
- Functions/methods used: `self.processor.process_video_file`, `np.zeros`, `astype`, `np.stack`, `np.asarray`
- Flow summary: `self.processor.process_video_file` -> `np.zeros` -> `astype` -> `np.stack` -> `np.asarray`

### Block: `VideoDataset._load_entry_arrays`
- Location: line 503
- Why this block exists: Loads configuration/state/data required by later runtime blocks.
- Variables (inputs) and roles:
  - `entry`: local value used by this block
- Variables created in this block: `key`, `data`, `skeleton_source`
- Functions/methods used: `entry.get`, `_safe_text`, `self._load_from_skeleton_source`, `print`, `self._load_preprocessed`, `self._load_from_video`
- Flow summary: `entry.get` -> `_safe_text` -> `self._load_from_skeleton_source` -> `print` -> `self._load_preprocessed` -> `self._load_from_video`

### Block: `VideoDataset._normalize_landmarks`
- Location: line 533
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `xyz`: input tensor/features
  - `mask`: validity mask tensor
- Variables created in this block: `xyz`, `mask`, `T`, `J`, `_`, `j`, `valid`, `c`, `l_hip`, `r_hip`, `hips_valid`, `hip_center`, `last`, `t`
- Functions/methods used: `xyz.copy`, `mask.copy`, `range`, `_fill_missing_1d`, `np.zeros`, `hips_valid.any`, `copy`, `np.where`, `np.ones`, `sh_valid.any`, `np.linalg.norm`, `np.clip`
- Flow summary: `xyz.copy` -> `mask.copy` -> `range` -> `_fill_missing_1d` -> `np.zeros` -> `hips_valid.any`

### Block: `VideoDataset._build_motion_features`
- Location: line 588
- Why this block exists: Constructs reusable objects/components from configuration.
- Variables (inputs) and roles:
  - `xyz`: input tensor/features
  - `mask`: validity mask tensor
- Variables created in this block: `vel`, `acc`, `feat`
- Functions/methods used: `np.zeros_like`, `astype`, `np.concatenate`
- Flow summary: `np.zeros_like` -> `astype` -> `np.concatenate`

### Block: `VideoDataset._sample_starts`
- Location: line 600
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `T`: local value used by this block
  - `window_size`: local value used by this block
  - `n_windows`: local value used by this block
- Variables created in this block: `max_start`
- Functions/methods used: `range`, `random.randint`, `tolist`, `astype`, `np.linspace`
- Flow summary: `range` -> `random.randint` -> `tolist` -> `astype` -> `np.linspace`

### Block: `VideoDataset.__getitem__`
- Location: line 610
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `idx`: local value used by this block
- Variables created in this block: `entry`, `data`, `xyz`, `mask`, `timestamps`, `quality`, `motion`, `window_size`, `n_windows`, `starts`, `windows`, `masks`, `win_timestamps`, `s`
- Functions/methods used: `self._load_entry_arrays`, `data.get`, `self._normalize_landmarks`, `self._build_motion_features`, `random.choice`, `int`, `max`, `self._sample_starts`, `np.pad`, `windows.append`, `w.astype`, `masks.append`
- Flow summary: `self._load_entry_arrays` -> `data.get` -> `self._normalize_landmarks` -> `self._build_motion_features` -> `random.choice` -> `int`

### Block: `collate_motion_batch`
- Location: line 691
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `batch`: single batch payload from loader
- Variables created in this block: `max_w`, `s`, `j`, `f`, `motion_list`, `mask_list`, `ts_list`, `quality`, `labels`, `video_ids`, `subject_ids`, `window_sizes`, `action_types`, `action_ids`
- Functions/methods used: `max`, `int`, `torch.cat`, `torch.zeros`, `motion_list.append`, `mask_list.append`, `ts_list.append`, `labels.append`, `action_types.append`, `item.get`, `action_ids.append`, `torch.tensor`
- Flow summary: `max` -> `int` -> `torch.cat` -> `torch.zeros` -> `motion_list.append` -> `mask_list.append`

## File: `src/training/logging_utils.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `json`, `os`, `time`, `matplotlib.pyplot`, `numpy`, `matplotlib.backends.backend_pdf: PdfPages`

### Class: `ExperimentLogger`
- Location: line 10
- Why this class exists: groups related state + methods into one reusable component.

### Block: `ExperimentLogger.__init__`
- Location: line 11
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `out_path`: module/model output dictionary
- Variables created in this block: `directory`
- Functions/methods used: `str`, `os.path.dirname`, `os.makedirs`
- Flow summary: `str` -> `os.path.dirname` -> `os.makedirs`

### Block: `ExperimentLogger.log`
- Location: line 17
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `stage`: local value used by this block
- Variables created in this block: `payload`
- Functions/methods used: `int`, `time.time`, `str`, `payload.update`, `open`, `f.write`, `json.dumps`
- Flow summary: `int` -> `time.time` -> `str` -> `payload.update` -> `open` -> `f.write`

### Block: `_read_jsonl`
- Location: line 27
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `path`: filesystem path string
- Variables created in this block: `rows`, `line`
- Functions/methods used: `os.path.exists`, `open`, `line.strip`, `rows.append`, `json.loads`
- Flow summary: `os.path.exists` -> `open` -> `line.strip` -> `rows.append` -> `json.loads`

### Block: `_group_by_stage`
- Location: line 43
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `rows`: local value used by this block
- Variables created in this block: `out`, `r`, `stage`
- Functions/methods used: `str`, `r.get`, `append`, `out.setdefault`
- Flow summary: `str` -> `r.get` -> `append` -> `out.setdefault`

### Block: `_extract_series`
- Location: line 51
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `rows`: local value used by this block
  - `x_key`: input tensor/features
  - `y_key`: target labels/targets
- Variables created in this block: `xs`, `ys`, `r`, `x`, `y`
- Functions/methods used: `float`, `np.isfinite`, `xs.append`, `ys.append`, `np.asarray`
- Flow summary: `float` -> `np.isfinite` -> `xs.append` -> `ys.append` -> `np.asarray`

### Block: `export_experiment_log_pdf`
- Location: line 68
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `log_jsonl_path`: local value used by this block
  - `pdf_path`: local value used by this block
  - `title`: local value used by this block
  - `extra_summary`: local value used by this block
- Variables created in this block: `rows`, `stages`, `fig`, `ax`, `stage_rows`, `table`, `summary_rows`, `table2`, `ssl_rows`, `action_rows`, `axes`, `x`, `y`, `x1`
- Functions/methods used: `_read_jsonl`, `_group_by_stage`, `os.makedirs`, `os.path.dirname`, `PdfPages`, `plt.figure`, `fig.add_axes`, `ax.axis`, `ax.text`, `time.strftime`, `os.path.basename`, `len`
- Flow summary: `_read_jsonl` -> `_group_by_stage` -> `os.makedirs` -> `os.path.dirname` -> `PdfPages` -> `plt.figure`

## File: `src/training/losses.py`
- Module intent: Loss functions for ASD Pipeline training.
- Key dependencies: `torch`, `torch.nn`, `torch.nn.functional`

### Class: `WeightedBCELoss`
- Location: line 14
- Why this class exists: groups related state + methods into one reusable component.

### Block: `WeightedBCELoss.__init__`
- Location: line 23
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `pos_weight`: local value used by this block
  - `label_smoothing`: local value used by this block
  - `logit_clip`: local value used by this block
  - `brier_weight`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `__init__`, `super`, `self.register_buffer`, `torch.tensor`, `float`, `max`, `min`
- Flow summary: `__init__` -> `super` -> `self.register_buffer` -> `torch.tensor` -> `float` -> `max`

### Block: `WeightedBCELoss.compute_from_labels`
- Location: line 41
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `labels`: target labels
  - `pos_weight_cap`: local value used by this block
- Variables created in this block: `labels`, `n_pos`, `n_neg`
- Functions/methods used: `isinstance`, `labels.numpy`, `np.asarray`, `max`, `labels.sum`, `len`, `min`, `float`
- Flow summary: `isinstance` -> `labels.numpy` -> `np.asarray` -> `max` -> `labels.sum` -> `len`

### Block: `WeightedBCELoss.forward`
- Location: line 52
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `logits`: raw model outputs before sigmoid/softmax
  - `target`: target labels
- Variables created in this block: `logits`, `target_raw`, `target`, `pw`, `bce`, `probs`, `brier`
- Functions/methods used: `logits.clamp`, `target.float`, `torch.tensor`, `pw.to`, `F.binary_cross_entropy_with_logits`, `torch.sigmoid`, `F.mse_loss`
- Flow summary: `logits.clamp` -> `target.float` -> `torch.tensor` -> `pw.to` -> `F.binary_cross_entropy_with_logits` -> `torch.sigmoid`

### Block: `pairwise_auc_loss`
- Location: line 83
- Why this block exists: Computes a scalar optimization objective from predictions and targets.
- Variables (inputs) and roles:
  - `logits`: raw model outputs before sigmoid/softmax
  - `target`: target labels
  - `temperature`: local value used by this block
- Variables created in this block: `logits`, `target`, `pos`, `neg`, `tau`, `diff`
- Functions/methods used: `view`, `logits.float`, `target.float`, `pos.numel`, `neg.numel`, `logits.new_tensor`, `max`, `float`, `pos.unsqueeze`, `neg.unsqueeze`, `mean`, `F.softplus`
- Flow summary: `view` -> `logits.float` -> `target.float` -> `pos.numel` -> `neg.numel` -> `logits.new_tensor`

### Block: `event_gate_bag_loss`
- Location: line 101
- Why this block exists: Computes a scalar optimization objective from predictions and targets.
- Variables (inputs) and roles:
  - `frame_event_scores`: local value used by this block
  - `target`: target labels
  - `frame_valid_mask`: local value used by this block
  - `eps`: local value used by this block
- Variables created in this block: `scores`, `valid`, `p`, `log_no_event_frame`, `log_no_event_window`, `no_event_window`, `valid_window`, `log_no_event_video`, `video_event_prob`, `y`
- Functions/methods used: `frame_event_scores.float`, `scores.dim`, `scores.unsqueeze`, `ValueError`, `tuple`, `torch.ones_like`, `frame_valid_mask.bool`, `valid.dim`, `valid.unsqueeze`, `scores.clamp`, `float`, `torch.log1p`
- Flow summary: `frame_event_scores.float` -> `scores.dim` -> `scores.unsqueeze` -> `ValueError` -> `tuple` -> `torch.ones_like`

### Block: `sens_at_spec_surrogate`
- Location: line 148
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `logits`: raw model outputs before sigmoid/softmax
  - `target`: target labels
  - `target_spec`: target labels
  - `margin`: local value used by this block
  - `detach_threshold`: local value used by this block
- Variables created in this block: `logits`, `target`, `probs`, `pos`, `neg`, `q`, `thr`, `m`, `pos_loss`, `neg_loss`
- Functions/methods used: `view`, `logits.float`, `target.float`, `torch.sigmoid`, `logits.clamp`, `pos.numel`, `neg.numel`, `probs.new_tensor`, `min`, `max`, `float`, `torch.quantile`
- Flow summary: `view` -> `logits.float` -> `target.float` -> `torch.sigmoid` -> `logits.clamp` -> `pos.numel`

### Block: `nas_entropy_regularization`
- Location: line 182
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `controller`: local value used by this block
  - `weight`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `controller.arch_entropy_loss`
- Flow summary: `controller.arch_entropy_loss`

## File: `src/training/motion_ssl.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `math`, `time`, `torch`, `torch.nn`, `torch.nn.functional`

### Block: `_interpolate_time`
- Location: line 9
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
  - `out_t`: module/model output dictionary
- Variables created in this block: `n`, `t`, `j`, `f`, `xf`, `y`
- Functions/methods used: `reshape`, `x.permute`, `F.interpolate`, `contiguous`, `permute`, `y.reshape`
- Flow summary: `reshape` -> `x.permute` -> `F.interpolate` -> `contiguous` -> `permute` -> `y.reshape`

### Block: `augment_motion_windows`
- Location: line 18
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
  - `joint_mask`: local value used by this block
  - `joint_dropout`: local value used by this block
  - `coord_noise_std`: local value used by this block
  - `temporal_mask_ratio`: local value used by this block
  - `speed_min`: local value used by this block
  - `speed_max`: local value used by this block
- Variables created in this block: `y`, `n`, `w`, `j`, `_`, `device`, `jd`, `joint_mask`, `noise`, `n_mask`, `i`, `start`, `speed`, `target_t`
- Functions/methods used: `x.clone`, `float`, `torch.rand`, `unsqueeze`, `jd.unsqueeze`, `jd.squeeze`, `torch.randn_like`, `max`, `int`, `range`, `item`, `torch.randint`
- Flow summary: `x.clone` -> `float` -> `torch.rand` -> `unsqueeze` -> `jd.unsqueeze` -> `jd.squeeze`

### Block: `info_nce`
- Location: line 71
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `z1`: local value used by this block
  - `z2`: local value used by this block
  - `temperature`: local value used by this block
- Variables created in this block: `z1`, `z2`, `logits`, `labels`
- Functions/methods used: `F.normalize`, `max`, `float`, `torch.arange`, `F.cross_entropy`
- Flow summary: `F.normalize` -> `max` -> `float` -> `torch.arange` -> `F.cross_entropy`

### Block: `pretrain_motion_encoder`
- Location: line 79
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `model`: model instance
  - `loader`: data loader iterator
  - `device`: runtime compute device (cpu/cuda)
  - `epochs`: training epoch counter
  - `lr`: local value used by this block
  - `max_steps_per_epoch`: local value used by this block
  - `logger`: logging utility/object
- Variables created in this block: `p`, `proj_dim`, `predictor`, `params`, `optimizer`, `scaler`, `epoch`, `epoch_loss`, `n_steps`, `started`, `batch`, `windows`, `joint_mask`, `b`
- Functions/methods used: `model.train`, `model.motion_encoder.parameters`, `to`, `nn.Sequential`, `nn.Linear`, `nn.GELU`, `list`, `predictor.parameters`, `torch.optim.AdamW`, `float`, `torch.amp.GradScaler`, `str`
- Flow summary: `model.train` -> `model.motion_encoder.parameters` -> `to` -> `nn.Sequential` -> `nn.Linear` -> `nn.GELU`

## File: `src/training/motion_ssl_landmark_augment.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `torch`, `torch.nn.functional`

### Block: `_resample_time`
- Location: line 5
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
  - `out_t`: module/model output dictionary
- Variables created in this block: `b`, `t`, `j`, `f`, `h`, `y`
- Functions/methods used: `reshape`, `x.permute`, `F.interpolate`, `int`, `contiguous`, `permute`, `y.reshape`
- Flow summary: `reshape` -> `x.permute` -> `F.interpolate` -> `int` -> `contiguous` -> `permute`

### Class: `MotionAugmentationPipeline`
- Location: line 13
- Why this class exists: groups related state + methods into one reusable component.

### Block: `MotionAugmentationPipeline.__init__`
- Location: line 14
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `joint_dropout_prob`: local value used by this block
  - `coordinate_noise_std`: local value used by this block
  - `temporal_mask_ratio`: local value used by this block
  - `speed_min`: local value used by this block
  - `speed_max`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `float`
- Flow summary: `float`

### Block: `MotionAugmentationPipeline._joint_dropout`
- Location: line 28
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `b`, `_`, `j`, `drop_mask`
- Functions/methods used: `torch.rand`, `unsqueeze`, `drop_mask.unsqueeze`, `x.masked_fill`
- Flow summary: `torch.rand` -> `unsqueeze` -> `drop_mask.unsqueeze` -> `x.masked_fill`

### Block: `MotionAugmentationPipeline._coordinate_noise`
- Location: line 36
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `y`, `noise`
- Functions/methods used: `x.clone`, `torch.randn_like`
- Flow summary: `x.clone` -> `torch.randn_like`

### Block: `MotionAugmentationPipeline._temporal_mask`
- Location: line 44
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `y`, `b`, `t`, `_`, `mask_len`, `i`, `start`
- Functions/methods used: `x.clone`, `max`, `int`, `round`, `range`, `item`, `torch.randint`
- Flow summary: `x.clone` -> `max` -> `int` -> `round` -> `range` -> `item`

### Block: `MotionAugmentationPipeline._speed_perturb`
- Location: line 58
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `b`, `t`, `_`, `y`, `i`, `speed`, `target_t`, `z`
- Functions/methods used: `x.clone`, `range`, `item`, `uniform_`, `torch.empty`, `max`, `int`, `round`, `float`, `_resample_time`
- Flow summary: `x.clone` -> `range` -> `item` -> `uniform_` -> `torch.empty` -> `max`

### Block: `MotionAugmentationPipeline.__call__`
- Location: line 70
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `x`: input tensor/features
- Variables created in this block: `y`
- Functions/methods used: `self._joint_dropout`, `self._coordinate_noise`, `self._temporal_mask`, `self._speed_perturb`, `y.contiguous`
- Flow summary: `self._joint_dropout` -> `self._coordinate_noise` -> `self._temporal_mask` -> `self._speed_perturb` -> `y.contiguous`

### Block: `build_positive_pair`
- Location: line 79
- Why this block exists: Constructs reusable objects/components from configuration.
- Variables (inputs) and roles:
  - `x`: input tensor/features
  - `augmenter`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `augmenter`
- Flow summary: `augmenter`

## File: `src/training/motion_ssl_landmark_dataset.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `csv`, `os`, `random`, `numpy`, `torch`, `torch.utils.data: DataLoader, Dataset`, `src.models.video.mediapipe_layer.landmark_schema: DEFAULT_SCHEMA`

### Block: `_safe_text`
- Location: line 12
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `value`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `strip`, `str`
- Flow summary: `strip` -> `str`

### Class: `LandmarkMotionPretrainDataset`
- Location: line 18
- Why this class exists: groups related state + methods into one reusable component.

### Block: `LandmarkMotionPretrainDataset.__init__`
- Location: line 26
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `source_path`: local value used by this block
  - `window_length`: local value used by this block
  - `future_offsets`: local value used by this block
  - `samples_per_epoch`: local value used by this block
  - `expected_joints`: local value used by this block
  - `cache_enabled`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `int`, `tuple`, `sorted`, `max`, `bool`, `self._collect_files`, `str`, `ValueError`
- Flow summary: `int` -> `tuple` -> `sorted` -> `max` -> `bool` -> `self._collect_files`

### Block: `LandmarkMotionPretrainDataset._collect_files`
- Location: line 48
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `source_path`: local value used by this block
- Variables created in this block: `files`, `name`, `ext`, `base_dir`, `reader`, `row`, `key`, `value`, `full`, `line`, `p`
- Functions/methods used: `os.path.isdir`, `os.walk`, `lower`, `os.path.splitext`, `files.append`, `os.path.join`, `files.sort`, `os.path.dirname`, `os.path.abspath`, `open`, `csv.DictReader`, `_safe_text`
- Flow summary: `os.path.isdir` -> `os.walk` -> `lower` -> `os.path.splitext` -> `files.append` -> `os.path.join`

### Block: `LandmarkMotionPretrainDataset._parse_ntu_skeleton_file`
- Location: line 90
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `path`: filesystem path string
- Variables created in this block: `lines`, `p`, `n_frames`, `frames`, `max_joints`, `_`, `n_bodies`, `best`, `best_valid`, `n_joints`, `joints`, `vals`, `x`, `y`
- Functions/methods used: `open`, `ln.strip`, `ValueError`, `int`, `float`, `range`, `max`, `len`, `split`, `joints.append`, `np.asarray`, `sum`
- Flow summary: `open` -> `ln.strip` -> `ValueError` -> `int` -> `float` -> `range`

### Block: `LandmarkMotionPretrainDataset.__len__`
- Location: line 174
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles: none
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `len`
- Flow summary: `len`

### Block: `LandmarkMotionPretrainDataset._compute_motion_features`
- Location: line 180
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `xyz`: input tensor/features
- Variables created in this block: `vel`, `acc`
- Functions/methods used: `np.zeros_like`, `astype`, `np.concatenate`
- Flow summary: `np.zeros_like` -> `astype` -> `np.concatenate`

### Block: `LandmarkMotionPretrainDataset._normalize_shape`
- Location: line 188
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `arr`: local value used by this block
- Variables created in this block: `x`, `t`, `j`, `c`, `pad`, `pad_j`
- Functions/methods used: `np.asarray`, `ValueError`, `self._compute_motion_features`, `np.zeros`, `np.concatenate`, `np.pad`, `x.astype`
- Flow summary: `np.asarray` -> `ValueError` -> `self._compute_motion_features` -> `np.zeros` -> `np.concatenate` -> `np.pad`

### Block: `LandmarkMotionPretrainDataset._load_file`
- Location: line 209
- Why this block exists: Loads configuration/state/data required by later runtime blocks.
- Variables (inputs) and roles:
  - `path`: filesystem path string
- Variables created in this block: `ext`, `arr`, `loaded`, `obj`, `key`, `value`, `seq`
- Functions/methods used: `lower`, `os.path.splitext`, `np.load`, `isinstance`, `loaded.item`, `torch.load`, `loaded.values`, `torch.is_tensor`, `self._parse_ntu_skeleton_file`, `ValueError`, `numpy`, `float`
- Flow summary: `lower` -> `os.path.splitext` -> `np.load` -> `isinstance` -> `loaded.item` -> `torch.load`

### Block: `LandmarkMotionPretrainDataset._sample_pair`
- Location: line 264
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `seq`: local value used by this block
- Variables created in this block: `t`, `w`, `max_offset`, `min_frames`, `pad_t`, `seq`, `max_start`, `start`, `horizon`, `target_start`, `anchor`, `target`
- Functions/methods used: `int`, `max`, `np.pad`, `random.randint`, `random.choice`, `min`, `anchor.astype`, `target.astype`
- Flow summary: `int` -> `max` -> `np.pad` -> `random.randint` -> `random.choice` -> `min`

### Block: `LandmarkMotionPretrainDataset.__getitem__`
- Location: line 283
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `idx`: local value used by this block
- Variables created in this block: `file_idx`, `path`, `seq`, `anchor`, `target`, `horizon`
- Functions/methods used: `random.randint`, `len`, `int`, `self._load_file`, `self._sample_pair`, `torch.from_numpy`, `torch.tensor`
- Flow summary: `random.randint` -> `len` -> `int` -> `self._load_file` -> `self._sample_pair` -> `torch.from_numpy`

### Block: `collate_landmark_ssl_batch`
- Location: line 300
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `batch`: single batch payload from loader
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `torch.stack`
- Flow summary: `torch.stack`

### Block: `build_landmark_ssl_dataloader`
- Location: line 310
- Why this block exists: Constructs reusable objects/components from configuration.
- Variables (inputs) and roles:
  - `dataset`: dataset object
  - `batch_size`: single batch payload from loader
  - `num_workers`: local value used by this block
  - `pin_memory`: local value used by this block
  - `shuffle`: local value used by this block
  - `drop_last`: local value used by this block
- Variables created in this block: `num_workers`, `kwargs`
- Functions/methods used: `int`, `max`, `bool`, `DataLoader`
- Flow summary: `int` -> `max` -> `bool` -> `DataLoader`

## File: `src/training/motion_ssl_landmark_losses.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `torch`, `torch.nn`, `torch.nn.functional`

### Block: `temporal_contrastive_infonce`
- Location: line 6
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `z1`: local value used by this block
  - `z2`: local value used by this block
  - `temperature`: local value used by this block
- Variables created in this block: `t`, `z1`, `z2`, `logits_12`, `logits_21`, `labels`, `loss`
- Functions/methods used: `max`, `float`, `F.normalize`, `z2.transpose`, `z1.transpose`, `torch.arange`, `F.cross_entropy`
- Flow summary: `max` -> `float` -> `F.normalize` -> `z2.transpose` -> `z1.transpose` -> `torch.arange`

### Class: `FutureMotionPredictor`
- Location: line 22
- Why this class exists: groups related state + methods into one reusable component.

### Block: `FutureMotionPredictor.__init__`
- Location: line 23
- Why this block exists: Initializes module state, submodules, and default hyperparameters.
- Variables (inputs) and roles:
  - `embedding_dim`: local value used by this block
  - `hidden_dim`: local value used by this block
  - `max_horizon`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `__init__`, `super`, `nn.Embedding`, `int`, `nn.Sequential`, `nn.Linear`, `nn.GELU`
- Flow summary: `__init__` -> `super` -> `nn.Embedding` -> `int` -> `nn.Sequential` -> `nn.Linear`

### Block: `FutureMotionPredictor.forward`
- Location: line 32
- Why this block exists: Executes the main data/tensor flow for this component.
- Variables (inputs) and roles:
  - `anchor_embedding`: local value used by this block
  - `horizon`: local value used by this block
- Variables created in this block: `h`, `h_emb`, `x`
- Functions/methods used: `horizon.clamp`, `self.horizon_emb`, `torch.cat`, `self.net`
- Flow summary: `horizon.clamp` -> `self.horizon_emb` -> `torch.cat` -> `self.net`

### Block: `future_motion_prediction_loss`
- Location: line 39
- Why this block exists: Computes a scalar optimization objective from predictions and targets.
- Variables (inputs) and roles:
  - `predicted`: local value used by this block
  - `target`: target labels
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `F.smooth_l1_loss`, `target.detach`
- Flow summary: `F.smooth_l1_loss` -> `target.detach`

## File: `src/training/nas_search.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `copy`, `time`, `numpy`, `torch`, `torch.utils.data: DataLoader, Subset`, `src.models.nas_controller: MicroGeneticNAS, default_search_space`, `src.models.pipeline_model: ASDPipeline`, `src.training.dataset: collate_motion_batch`, `src.training.losses: WeightedBCELoss, event_gate_bag_loss`, `src.utils.metrics: compute_auc, compute_ece, sensitivity_at_specificity`, `src.utils.splits: make_group_kfold, check_group_overlap`

### Block: `_to_device`
- Location: line 16
- Why this block exists: Converts/transforms structures into the format expected downstream.
- Variables (inputs) and roles:
  - `batch`: single batch payload from loader
  - `device`: runtime compute device (cpu/cuda)
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `to`
- Flow summary: `to`

### Block: `_train_epoch`
- Location: line 24
- Why this block exists: Runs training logic and returns optimization/training signals.
- Variables (inputs) and roles:
  - `model`: model instance
  - `loader`: data loader iterator
  - `optimizer`: parameter optimizer
  - `scaler`: local value used by this block
  - `criterion`: local value used by this block
  - `device`: runtime compute device (cpu/cuda)
  - `gate_aux_weight`: local value used by this block
- Variables created in this block: `total`, `n`, `batch`, `y`, `x`, `out`, `cls_loss`, `gate_aux`, `loss`
- Functions/methods used: `model.train`, `to`, `_to_device`, `torch.amp.autocast`, `startswith`, `str`, `model`, `criterion`, `cls_loss.new_tensor`, `float`, `out.get`, `event_gate_bag_loss`
- Flow summary: `model.train` -> `to` -> `_to_device` -> `torch.amp.autocast` -> `startswith` -> `str`

### Block: `_evaluate`
- Location: line 55
- Why this block exists: Runs evaluation/inference logic and returns metrics/predictions.
- Variables (inputs) and roles:
  - `model`: model instance
  - `loader`: data loader iterator
  - `device`: runtime compute device (cpu/cuda)
- Variables created in this block: `probs`, `labels`, `started`, `batch`, `y`, `x`, `out`, `p`, `infer_time`, `auc`, `sens90`, `ece`
- Functions/methods used: `model.eval`, `time.time`, `to`, `_to_device`, `model`, `torch.sigmoid`, `probs.append`, `numpy`, `cpu`, `p.detach`, `labels.append`, `y.detach`
- Flow summary: `model.eval` -> `time.time` -> `to` -> `_to_device` -> `model` -> `torch.sigmoid`

### Block: `_efficiency_penalty`
- Location: line 84
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `train_time`: local value used by this block
  - `infer_time`: local value used by this block
  - `max_mem_gb`: local value used by this block
- Variables created in this block: `t_pen`, `i_pen`, `m_pen`
- Functions/methods used: `np.clip`, `float`
- Flow summary: `np.clip` -> `float`

### Block: `run_micro_genetic_nas`
- Location: line 92
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `cfg`: configuration dictionary
  - `dataset`: dataset object
  - `labels`: target labels
  - `groups`: local value used by this block
  - `device`: runtime compute device (cpu/cuda)
  - `logger`: logging utility/object
- Variables created in this block: `nas_cfg`, `t_cfg`, `data_cfg`, `model_cfg`, `threshold_cfg`, `pop`, `gens`, `tour`, `mut`, `crossover`, `elite`, `nas_epochs`, `nas_folds`, `gate_aux_weight`
- Functions/methods used: `cfg.get`, `int`, `nas_cfg.get`, `float`, `bool`, `t_cfg.get`, `data_cfg.get`, `copy.deepcopy`, `default_search_space`, `make_group_kfold`, `max`, `min`
- Flow summary: `cfg.get` -> `int` -> `nas_cfg.get` -> `float` -> `bool` -> `t_cfg.get`

## File: `src/training/train.py`
- Module intent: no top-level docstring; inferred from symbol names below.
- Key dependencies: `argparse`, `json`, `os`, `time`, `numpy`, `torch`, `torch.utils.data: DataLoader, Subset`, `src.models.pipeline_model: ASDPipeline`, `src.pipeline.preprocess: precompute_videos`, `src.training.checkpoints: CheckpointManager`, `src.training.dataset: VideoDataset, collate_motion_batch`, `src.training.logging_utils: ExperimentLogger, export_experiment_log_pdf`, `src.training.losses: WeightedBCELoss, event_gate_bag_loss`, `src.training.motion_ssl: pretrain_motion_encoder`, `src.training.nas_search: run_micro_genetic_nas`, `src.utils.calibration: apply_temperature, fit_temperature`, `src.utils.config: apply_overrides, load_config`, `src.utils.metrics: compute_auc, compute_basic_metrics, compute_ece, find_optimal_threshold, sensitivity_at_specificity`, `src.utils.seed: seed_everything, seed_worker`, `src.utils.splits: check_group_overlap, make_group_kfold, make_group_stratified_split`

### Block: `_write_status_file`
- Location: line 31
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `status_file`: local value used by this block
  - `payload`: local value used by this block
- Variables created in this block: `directory`, `tmp`
- Functions/methods used: `os.path.dirname`, `os.makedirs`, `open`, `json.dump`, `os.replace`
- Flow summary: `os.path.dirname` -> `os.makedirs` -> `open` -> `json.dump` -> `os.replace`

### Block: `_is_rtx_4050`
- Location: line 43
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `device`: runtime compute device (cpu/cuda)
- Variables created in this block: `props`
- Functions/methods used: `startswith`, `str`, `torch.cuda.is_available`, `torch.cuda.get_device_properties`, `lower`
- Flow summary: `startswith` -> `str` -> `torch.cuda.is_available` -> `torch.cuda.get_device_properties` -> `lower`

### Block: `_auto_batch_and_workers`
- Location: line 53
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `cfg`: configuration dictionary
  - `device`: runtime compute device (cpu/cuda)
- Variables created in this block: `train_cfg`, `data_cfg`
- Functions/methods used: `cfg.setdefault`, `_is_rtx_4050`, `int`
- Flow summary: `cfg.setdefault` -> `_is_rtx_4050` -> `int`

### Block: `_build_dataset`
- Location: line 64
- Why this block exists: Constructs reusable objects/components from configuration.
- Variables (inputs) and roles:
  - `cfg`: configuration dictionary
  - `is_training`: local value used by this block
- Variables created in this block: `data_cfg`
- Functions/methods used: `cfg.get`, `VideoDataset`, `str`, `data_cfg.get`, `int`, `bool`, `tuple`
- Flow summary: `cfg.get` -> `VideoDataset` -> `str` -> `data_cfg.get` -> `int` -> `bool`

### Block: `_build_loader`
- Location: line 82
- Why this block exists: Constructs reusable objects/components from configuration.
- Variables (inputs) and roles:
  - `dataset`: dataset object
  - `cfg`: configuration dictionary
  - `shuffle`: local value used by this block
  - `generator`: local value used by this block
- Variables created in this block: `data_cfg`, `train_cfg`, `num_workers`, `kwargs`
- Functions/methods used: `cfg.get`, `int`, `data_cfg.get`, `train_cfg.get`, `bool`, `DataLoader`
- Flow summary: `cfg.get` -> `int` -> `data_cfg.get` -> `train_cfg.get` -> `bool` -> `DataLoader`

### Block: `_to_inputs`
- Location: line 101
- Why this block exists: Converts/transforms structures into the format expected downstream.
- Variables (inputs) and roles:
  - `batch`: single batch payload from loader
  - `device`: runtime compute device (cpu/cuda)
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `to`
- Flow summary: `to`

### Block: `_quality_score_from_batch`
- Location: line 109
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `batch`: single batch payload from loader
- Variables created in this block: `q`, `face`, `pose`, `hand`, `score`
- Functions/methods used: `float`, `score.clamp`
- Flow summary: `float` -> `score.clamp`

### Block: `train_one_epoch`
- Location: line 118
- Why this block exists: Runs training logic and returns optimization/training signals.
- Variables (inputs) and roles:
  - `model`: model instance
  - `loader`: data loader iterator
  - `criterion`: local value used by this block
  - `optimizer`: parameter optimizer
  - `scaler`: local value used by this block
  - `device`: runtime compute device (cpu/cuda)
  - `clip_grad`: local value used by this block
  - `gate_aux_weight`: local value used by this block
- Variables created in this block: `total`, `n`, `batch`, `labels`, `inputs`, `out`, `cls_loss`, `gate_aux`, `loss`
- Functions/methods used: `model.train`, `to`, `_to_inputs`, `torch.amp.autocast`, `startswith`, `str`, `model`, `criterion`, `cls_loss.new_tensor`, `float`, `out.get`, `event_gate_bag_loss`
- Flow summary: `model.train` -> `to` -> `_to_inputs` -> `torch.amp.autocast` -> `startswith` -> `str`

### Block: `evaluate`
- Location: line 158
- Why this block exists: Runs evaluation/inference logic and returns metrics/predictions.
- Variables (inputs) and roles:
  - `model`: model instance
  - `loader`: data loader iterator
  - `criterion`: local value used by this block
  - `device`: runtime compute device (cpu/cuda)
- Variables created in this block: `total`, `n`, `logits`, `labels`, `qualities`, `batch`, `y`, `x`, `out`, `loss`, `logit`, `probs`
- Functions/methods used: `model.eval`, `to`, `_to_inputs`, `torch.amp.autocast`, `startswith`, `str`, `model`, `criterion`, `float`, `loss.item`, `logits.append`, `numpy`
- Flow summary: `model.eval` -> `to` -> `_to_inputs` -> `torch.amp.autocast` -> `startswith` -> `str`

### Block: `summarize_metrics`
- Location: line 199
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `labels`: target labels
  - `probs`: probability outputs
  - `spec_target`: local value used by this block
- Variables created in this block: `auc`, `ece`, `cal_quality`, `sens`, `thr`, `basic`
- Functions/methods used: `compute_auc`, `compute_ece`, `float`, `np.clip`, `sensitivity_at_specificity`, `np.isfinite`, `find_optimal_threshold`, `compute_basic_metrics`, `max`
- Flow summary: `compute_auc` -> `compute_ece` -> `float` -> `np.clip` -> `sensitivity_at_specificity` -> `np.isfinite`

### Block: `_selection_score`
- Location: line 232
- Why this block exists: Implements a supporting block used by the module workflow.
- Variables (inputs) and roles:
  - `metrics`: local value used by this block
- Variables created in this block: none or delegated to child calls
- Functions/methods used: `float`, `metrics.get`
- Flow summary: `float` -> `metrics.get`

### Block: `_build_model_from_cfg`
- Location: line 242
- Why this block exists: Constructs reusable objects/components from configuration.
- Variables (inputs) and roles:
  - `cfg`: configuration dictionary
- Variables created in this block: `model_cfg`, `thresholds`
- Functions/methods used: `cfg.get`, `ASDPipeline`, `int`, `model_cfg.get`, `float`, `thresholds.get`
- Flow summary: `cfg.get` -> `ASDPipeline` -> `int` -> `model_cfg.get` -> `float` -> `thresholds.get`

### Block: `_load_pretrained_motion_encoder`
- Location: line 254
- Why this block exists: Loads configuration/state/data required by later runtime blocks.
- Variables (inputs) and roles:
  - `model`: model instance
  - `checkpoint_path`: local value used by this block
  - `device`: runtime compute device (cpu/cuda)
- Variables created in this block: `ckpt`, `state`, `ms`, `missing`, `unexpected`
- Functions/methods used: `os.path.exists`, `print`, `torch.load`, `isinstance`, `ms.items`, `startswith`, `str`, `replace`, `model.motion_encoder.load_state_dict`, `len`
- Flow summary: `os.path.exists` -> `print` -> `torch.load` -> `isinstance` -> `ms.items` -> `startswith`

### Block: `train`
- Location: line 283
- Why this block exists: Runs training logic and returns optimization/training signals.
- Variables (inputs) and roles:
  - `cfg`: configuration dictionary
  - `status_file`: local value used by this block
- Variables created in this block: `seed`, `generator`, `device_name`, `device`, `results_dir`, `ckpt_mgr`, `run_tag`, `log_path`, `logger`, `started`, `payload`, `data_cfg`, `train_cfg`, `nas_cfg`
- Functions/methods used: `int`, `cfg.get`, `seed_everything`, `torch.device`, `torch.cuda.is_available`, `print`, `_auto_batch_and_workers`, `get`, `os.makedirs`, `CheckpointManager`, `time.strftime`, `os.path.join`
- Flow summary: `int` -> `cfg.get` -> `seed_everything` -> `torch.device` -> `torch.cuda.is_available` -> `print`

### Block: `main`
- Location: line 662
- Why this block exists: Script entrypoint that parses arguments and executes the workflow.
- Variables (inputs) and roles: none
- Variables created in this block: `parser`, `args`, `cfg`
- Functions/methods used: `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `load_config`, `args.override.append`, `apply_overrides`, `train`
- Flow summary: `argparse.ArgumentParser` -> `parser.add_argument` -> `parser.parse_args` -> `load_config` -> `args.override.append` -> `apply_overrides`
