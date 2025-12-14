# Search R1 Self Evolution - Development Notes

## Current Issues and Debugging

### H20 GPU SIGFPE Error (2025-12-14)

**Problem:**
- Getting SIGFPE (Floating Point Exception) when running on H20 GPUs
- Error occurs in the embedding layer during `generate_sequences()` call
- Stack trace shows failure in `vocab_parallel_embedding.py` -> `logits_processor.py`

**Error Details:**
```
Fatal Python error: Floating point exception
File "vllm/model_executor/layers/vocab_parallel_embedding.py", line 40 in apply
File "vllm/model_executor/layers/logits_processor.py", line 83 in _get_logits
```

**Root Cause Analysis:**
- H20 GPUs may be more sensitive to non-contiguous tensor memory layouts
- Hardware-specific numerical instabilities in certain operations
- NOT a dtype issue - token IDs must remain as `long` type for embeddings

**Solution Implemented:**
Modified `search_r1/llm_agent/generation.py` to ensure tensors are contiguous:

1. **Line 999**: Added `.contiguous()` after `.long()` for main batch
   ```python
   active_batch.batch[key] = active_batch.batch[key].long().contiguous()
   ```

2. **Line 1023**: Added `.contiguous()` for padded batch
   ```python
   padded_active_batch.batch[key] = padded_active_batch.batch[key].long().contiguous()
   ```

**Why This Might Help:**
- Ensures tensors have contiguous memory layout
- Improves hardware compatibility, especially for H20 GPUs
- Eliminates potential memory stride issues in low-level operations

**Testing Status:**
- ⏳ Awaiting test results on H20 GPU

**Fallback Options if Issue Persists:**
1. Verify CUDA/PyTorch version compatibility with H20 GPUs
2. Check for known vLLM issues with H20 hardware
3. Test with reduced batch size (potential memory corruption)
4. Consider upgrading vLLM to latest version with H20 support

---

## Code Structure Notes

### Key Files
- `search_r1/llm_agent/generation.py`: Main generation logic with GPU padding support
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`: vLLM rollout worker
- `verl/trainer/ppo/ray_trainer.py`: PPO training coordinator

### Modified Files (Current Session)
- `search_r1/llm_agent/generation.py`: Added `.contiguous()` for H20 GPU compatibility

---

## Git Status
Current branch: `main`
Modified files:
- memory_db/train/stage_rl/Memory_db/database.py
- search_r1/llm_agent/generation.py
- train_grpo.sh
- train_grpo_1gpu.sh
- verl/trainer/config/ppo_trainer.yaml
- verl/trainer/ppo/ray_trainer.py

---

---

### DataProto Concatenation AssertionError (2025-12-14)

**Problem:**
- AssertionError when concatenating `gen_batch_output` with `memory_db_dataproto`
- Error in `verl/protocol.py:99` during `list_of_dict_to_dict_of_list`
- Root cause: `non_tensor_batch` dictionaries must have matching keys for concatenation

**Error Details:**
```
File "verl/trainer/ppo/ray_trainer.py", line 1776, in _augment_batch_with_memory_db_responses
    gen_batch_output = DataProto.concat([gen_batch_output, memory_db_dataproto])
File "verl/protocol.py", line 533, in concat
    non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
File "verl/protocol.py", line 99, in list_of_dict_to_dict_of_list
    assert key in output
AssertionError
```

**Solution Implemented:**
Modified `verl/trainer/ppo/ray_trainer.py`:

1. **Lines 1726-1740**: In `_create_dataproto_from_memory_db_responses()`, copy all keys from `reference_batch.non_tensor_batch` to ensure matching structure
   - Replicate first sample values for all existing keys
   - Add `from_memory_db=True` marker

2. **Lines 1787-1792**: In `_augment_batch_with_memory_db_responses()`, ensure `gen_batch_output.non_tensor_batch` has `from_memory_db` key before concatenation
   - Initialize with `False` for original samples

**Testing Status:**
- ⏳ Awaiting test results

**Additional Fixes:**
3. **Line 1821**: Fixed `dtype=object` requirement for `non_tensor_batch` arrays in placeholder creation
4. **Lines 1847-1860**: Added defensive checks and error messages for batch size verification
   - Ensures `batch` and `gen_batch_output` have matching sizes before return
   - Prevents silent failures in subsequent `union()` operations
5. **Lines 1640-1672**: Added `_mask_system_content_in_response()` to properly mask system-inserted content
   - Masks `<information>...</information>` blocks (search results)
   - Masks system error message: "My previous action is invalid..."
6. **Lines 1685-1755**: Updated `_create_dataproto_from_memory_db_responses()` to create proper `responses_with_info_mask` and `info_mask`
   - Ensures model trains on negative responses but NOT on system-inserted content
   - Maintains same masking behavior as normal generation flow

---

## Next Steps
1. Test the `.contiguous()` fix on H20 GPU
2. Test the DataProto concatenation fix for memory DB augmentation
3. Monitor for SIGFPE errors and AssertionErrors
4. If successful, commit changes with descriptive message
5. If unsuccessful, investigate further
