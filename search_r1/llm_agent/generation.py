from doctest import Example
import torch
import re
import time
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    enable_revision: bool = True  # Whether to perform self-revision
    """"
    system
    You are a helpful assistant.
    user
    Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: big little lies season 2 how many episodes?
    assistant

    # The response may includes instruction below from system (not model) in the middle to guide to the model. \n\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again. \n\n It is not part of the model output and we should avoid outputing it in revised solution. 
    """
    # Complete prompt template for analysis step
    # This shows the entire structure of what the LLM sees when analyzing a response
    # for debug search "Revision Output (2 Steps: Analysis and Revision)"

    analysis_full_prompt_template: str = """

You are an expert in analyzing search queries used by LLM agents to answer questions.

You will be given the question (with instructions) and the search queries from the agent's response (if any). Your task is ONLY to analyze the search strategy, NOT to answer the question.

Here is the question:

{original_prompt}

Here are the search queries from the original response:

{original_response}

Please analyze the search strategy above. Identify:
1) Whether the search queries are properly formatted (each query should be inside <search> and </search> tags without nesting or unclosed tags). If no searches were made, note this.
2) Whether the number of searches is reasonable. IMPORTANT: The system has strict limits on external searches. Use fewer, more targeted searches - each search should be highly informative.
3) What search queries were used and their effectiveness in finding relevant information (or what searches should have been made if none were found)
4) What queries are missing or could be improved
5) Whether the queries are specific enough or too broad

Guidelines:
- Keep response under 100 words total
- Use plain text, no formatting
- Be specific and actionable
- DO NOT attempt to answer the question - only analyze the search strategy

Provide your analysis:

"""

    # Separate components used in actual implementation
    # (The templates above show the complete structure, but the code uses these individual pieces)

    top_instruction_for_analysis: str = """

You are an expert in analyzing search queries used by LLM agents to answer questions.

You will be given the question (with instructions) and the search queries from the agent's response (if any). Your task is ONLY to analyze the search strategy, NOT to answer the question.

Here is the question:

"""

    analysis_prompt: str = """

Please analyze the search strategy above. Identify:
1) Whether the search queries are properly formatted (each query should be inside <search> and </search> tags without nesting or unclosed tags). If no searches were made, note this.
2) Whether the number of searches is reasonable. IMPORTANT: The system has strict limits on external searches. Use fewer, more targeted searches - each search should be highly informative.
3) What search queries were used and their effectiveness in finding relevant information (or what searches should have been made if none were found)
4) What queries are missing or could be improved
5) Whether the queries are specific enough or too broad

Guidelines:
- Keep response under 100 words total
- Use plain text, no formatting
- Be specific and actionable
- DO NOT attempt to answer the question - only analyze the search strategy

Provide your analysis:

"""

    # Two-step revision settings
    two_step_revision: bool = True  # Enable two-step revision (analysis first, then revision)

    # Complete prompt template for revision step (after analysis)
    # This shows the entire structure of what the LLM sees when revising based on analysis
    revision_with_analysis_full_prompt_template: str = """

You are an expert in improving how LLM agents answer questions using external search.

You will be given the question (with instructions), the search queries from the agent's original response, and an analysis. Based on these, please develop a revised solution with better search queries and reasoning.

Here is the question:

{original_prompt}

Here are the search queries from the original response:

{original_response}

Here is the analysis of the search queries:

{analysis}

Based on the analysis above, please develop a revised solution. IMPORTANT: Please try not to change the queries in the original response - only change queries that are inappropriate or have very vague/unclear phrasing. You may also improve reasoning steps or adjust the search strategy. Follow ALL the instructions and format requirements specified in the question above, including using <think>, <search>, <answer> tags.

"""

    # Complete prompt template for single-step revision (without separate analysis)
    revision_single_step_full_prompt_template: str = """

You are an expert in improving how LLM agents answer questions using external search.

You will be given the question (with instructions) and the search queries from the agent's original response. Please develop a revised solution with better search queries and reasoning.

Here is the question:

{original_prompt}

Here are the search queries from the original response:

{original_response}

Please develop a revised solution with better search queries and reasoning. IMPORTANT: Please try not to change the queries in the original response - only change queries that are inappropriate or have very vague/unclear phrasing. You may also improve reasoning steps or adjust the search strategy. Follow ALL the instructions and format requirements specified in the question above, including using <think>, <search>, <answer> tags.

"""

    # Separate components used in actual implementation
    # (The templates above show the complete structure, but the code uses these individual pieces)

    top_instruction_for_revision_with_analysis: str = """

You are an expert in improving how LLM agents answer questions using external search.

You will be given the question (with instructions), the search queries from the agent's original response, and an analysis. Based on these, please develop a revised solution with better search queries and reasoning.

Here is the question:

"""

    top_instruction_for_revision_single_step: str = """

You are an expert in improving how LLM agents answer questions using external search.

You will be given the question (with instructions) and the search queries from the agent's original response. Please develop a revised solution with better search queries and reasoning.

Here is the question:

"""

    previous_response_instruction: str = """

Here are the search queries from the original response:

"""

    analysis_label: str = """

Here is the analysis of the search queries:

"""

    revision_instruction_for_two_step: str = """

Based on the analysis above, please develop a revised solution. IMPORTANT: Please try not to change the queries in the original response - only change queries that are inappropriate or have very vague/unclear phrasing. You may also improve reasoning steps or adjust the search strategy. Follow ALL the instructions and format requirements specified in the question above, including using <think>, <search>, <answer> tags.

"""

    revision_instruction_for_single_step: str = """

Please develop a revised solution with better search queries and reasoning. IMPORTANT: Please try not to change the queries in the original response - only change queries that are inappropriate or have very vague/unclear phrasing. You may also improve reasoning steps or adjust the search strategy. Follow ALL the instructions and format requirements specified in the question above, including using <think>, <search>, <answer> tags.

"""

    enable_transfer_learning: bool = True  # Enable transfer learning between thinking processes from the same prompt
    transfer_use_revised: bool = True  # If True, transfer learning uses revised responses; if False, uses original responses

    # Complete prompt template for transfer learning
    # This shows the entire structure of what the LLM sees when doing transfer learning
    transfer_full_prompt_template: str = """

You are an expert in improving how LLM agents answer questions using external search.

You will be given the question (with instructions) and the search queries from multiple previous thinking processes for this question. Please analyze these search queries and select the best query approach to generate a new, improved thinking process.

Here is the question:

{original_prompt}

Previous thinking process 1 (search queries only):

{previous_response_1}

Previous thinking process 2 (search queries only):

{previous_response_2}

...

Above are search queries from multiple previous thinking processes for the same question. Please analyze these search queries and select the best query approach to generate a new, improved thinking process. Use fewer, more targeted searches - minimize the number of searches while maximizing their informativeness. Follow ALL the instructions and format requirements specified in the question above, including using <think>, <search>, <answer> tags.

"""

    # Separate components used in actual transfer learning implementation
    # (The template above shows the complete structure, but the code uses these individual pieces)

    top_instruction_for_transfer: str = """

You are an expert in improving how LLM agents answer questions using external search.

You will be given the question (with instructions) and the search queries from multiple previous thinking processes for this question. Please analyze these search queries and select the best query approach to generate a new, improved thinking process.

Here is the question:

"""

    transfer_prompt: str = """

Above are search queries from multiple previous thinking processes for the same question. Please analyze these search queries and select the best query approach to generate a new, improved thinking process. Use fewer, more targeted searches - minimize the number of searches while maximizing their informativeness. Follow ALL the instructions and format requirements specified in the question above, including using <think>, <search>, <answer> tags.

"""

    debug_llm_io: bool = True  # Print raw input/output of LLM generation for debugging

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        n_agent: int = 1,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.n_agent = n_agent  # Number of agent repetitions per prompt

        # For tracking epoch and batch index in prints
        self.current_epoch = None
        self.current_batch_idx = None
        self.current_phase = None  # 'initial', 'revision', 'transfer', or None

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        # Validate that prompt components match templates
        self._validate_prompt_templates()

    def _validate_prompt_templates(self):
        """
        Validate that the separate prompt components can reconstruct the template.
        This ensures the components stay in sync with the documented templates.
        """
        # Test placeholders
        test_prompt = "TEST_PROMPT"
        test_response = "TEST_RESPONSE"
        test_analysis = "TEST_ANALYSIS"

        # Validate analysis template
        expected_analysis = self.config.analysis_full_prompt_template.format(
            original_prompt=test_prompt,
            original_response=test_response
        )
        actual_analysis = (
            self.config.top_instruction_for_analysis +
            test_prompt +
            self.config.previous_response_instruction +
            test_response +
            self.config.analysis_prompt
        )

        if expected_analysis.strip() != actual_analysis.strip():
            print("\n" + "="*80)
            print("WARNING: Analysis prompt components don't match template!")
            print("="*80)
            print("Expected (from template):")
            print(expected_analysis[:500])
            print("\nActual (from components):")
            print(actual_analysis[:500])
            print("="*80 + "\n")

        # Validate revision with analysis template
        expected_revision = self.config.revision_with_analysis_full_prompt_template.format(
            original_prompt=test_prompt,
            original_response=test_response,
            analysis=test_analysis
        )
        actual_revision = (
            self.config.top_instruction_for_revision_with_analysis +
            test_prompt +
            self.config.previous_response_instruction +
            test_response +
            self.config.analysis_label +
            test_analysis +
            self.config.revision_instruction_for_two_step
        )

        if expected_revision.strip() != actual_revision.strip():
            print("\n" + "="*80)
            print("WARNING: Revision prompt components don't match template!")
            print("="*80)
            print("Expected (from template):")
            print(expected_revision[:500])
            print("\nActual (from components):")
            print(actual_revision[:500])
            print("="*80 + "\n")

        # Validate single-step revision template
        expected_single = self.config.revision_single_step_full_prompt_template.format(
            original_prompt=test_prompt,
            original_response=test_response
        )
        actual_single = (
            self.config.top_instruction_for_revision_single_step +
            test_prompt +
            self.config.previous_response_instruction +
            test_response +
            self.config.revision_instruction_for_single_step
        )

        if expected_single.strip() != actual_single.strip():
            print("\n" + "="*80)
            print("WARNING: Single-step revision prompt components don't match template!")
            print("="*80)
            print("Expected (from template):")
            print(expected_single[:500])
            print("\nActual (from components):")
            print(actual_single[:500])
            print("="*80 + "\n")

    def _strip_special_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Remove special tokens (eos_token, pad_token) from both ends of token_ids.
        This prevents <endoftext> and padding from appearing in the middle of concatenated prompts.
        """
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # Find the first non-special token (strip from beginning)
        start_idx = 0
        while start_idx < len(token_ids):
            token = token_ids[start_idx].item()
            if token != eos_token_id and token != pad_token_id:
                break
            start_idx += 1

        # Find the last non-special token (strip from end)
        end_idx = len(token_ids)
        while end_idx > start_idx:
            token = token_ids[end_idx - 1].item()
            if token != eos_token_id and token != pad_token_id:
                break
            end_idx -= 1

        return token_ids[start_idx:end_idx] if end_idx > start_idx else token_ids[:0]

    def _extract_search_queries_only(self, text: str) -> str:
        """
        Extract only <search>...</search> content from the response text.
        Filters out blocks where the content is just "and" (from system-generated correction messages).
        This significantly reduces context length during analysis and revision by removing
        lengthy <think>, <information>, and <answer> sections.

        Args:
            text: The original response text

        Returns:
            Text containing only the <search> sections with their tags (excluding system examples)
        """
        import re
        # Find all <search>...</search> blocks
        search_pattern = r'<search>.*?</search>'
        search_blocks = re.findall(search_pattern, text, flags=re.DOTALL)

        # Filter out blocks where content is just "and" (from system correction messages)
        filtered_blocks = []
        for block in search_blocks:
            # Extract content between tags
            content = re.search(r'<search>(.*?)</search>', block, flags=re.DOTALL)
            if content:
                content_text = content.group(1).strip()
                # Skip if content is just "and" or "query" (from system-generated instruction examples)
                if content_text not in ['and', 'query']:
                    filtered_blocks.append(block)

        # Join all search blocks with newlines
        if filtered_blocks:
            return '\n'.join(filtered_blocks)
        else:
            # If no search tags found, return a system note
            return "[SYSTEM NOTE: No search queries found in the original response]"

    def _truncate_sequence(self,
                           segments: List[torch.Tensor],
                           truncatable_index: int,
                           max_len: int,
                           keep_end: bool = True) -> torch.Tensor:
        """
        Truncate a sequence of segments to fit within max_len.

        The sequence is formed by concatenating all segments. If the total length
        exceeds max_len, the segment at truncatable_index is truncated to fit.

        Args:
            segments: List of tensor segments to concatenate
            truncatable_index: Index of the segment that can be truncated
            max_len: Maximum allowed length for the final sequence
            keep_end: If True, keep the end of truncatable segment (truncate from beginning);
                      If False, keep the beginning (truncate from end)

        Returns:
            Concatenated and possibly truncated tensor
        """
        full_sequence = torch.cat(segments, dim=0)

        if len(full_sequence) <= max_len:
            return full_sequence

        # Calculate total length of fixed (non-truncatable) segments
        fixed_len = sum(len(seg) for i, seg in enumerate(segments) if i != truncatable_index)
        available_for_truncatable = max_len - fixed_len

        if available_for_truncatable > 0:
            truncatable_seg = segments[truncatable_index]
            if len(truncatable_seg) > available_for_truncatable:
                if keep_end:
                    # Keep the end (most recent context)
                    truncated_seg = truncatable_seg[-available_for_truncatable:]
                else:
                    # Keep the beginning
                    truncated_seg = truncatable_seg[:available_for_truncatable]
            else:
                truncated_seg = truncatable_seg

            # Reconstruct with truncated segment
            new_segments = []
            for i, seg in enumerate(segments):
                if i == truncatable_index:
                    new_segments.append(truncated_seg)
                else:
                    new_segments.append(seg)
            return torch.cat(new_segments, dim=0)
        else:
            # If even the fixed segments exceed max_len, truncate from the beginning
            return full_sequence[-max_len:]

    def _concatenate_and_truncate(self,
                                   components: List[torch.Tensor],
                                   truncation_segments: List[torch.Tensor] = None,
                                   truncatable_index: int = 0,
                                   max_len: int = None,
                                   keep_end: bool = True,
                                   debug: bool = False,
                                   debug_label: str = "Prompt",
                                   debug_info: Dict[str, Any] = None,
                                   print_char_limit: int = 6000) -> torch.Tensor:
        """
        Concatenate components and truncate if needed, with optional debug printing.

        Args:
            components: List of tensor components to concatenate
            truncation_segments: Segments to pass to _truncate_sequence (defaults to components if None)
            truncatable_index: Index of the segment to truncate in truncation_segments
            max_len: Maximum allowed length (if None, no truncation is performed)
            keep_end: If True, keep end of truncatable segment
            debug: If True, print debug info before and after truncation
            debug_label: Label for debug prints (e.g., "Transfer Learning Prompt")
            debug_info: Optional dict of additional debug info to print
            print_char_limit: Character limit for decoded prompt printing

        Returns:
            Concatenated and possibly truncated tensor
        """
        # Use components as truncation_segments if not specified
        if truncation_segments is None:
            truncation_segments = components

        # Concatenate all components
        full_prompt = torch.cat(components, dim=0)

        # Debug: Print before truncation
        if debug:
            print(f"\n{'='*80}")
            print(f"DEBUG: {debug_label} Structure (BEFORE truncation)")
            print(f"{'='*80}")
            print(f"Full prompt length: {len(full_prompt)} tokens")
            if debug_info:
                for key, value in debug_info.items():
                    print(f"{key}: {value}")

            # Decode and print the full prompt
            full_prompt_decoded = self.tokenizer.decode(full_prompt.long(), skip_special_tokens=True)
            print(f"\nFull prompt content (first {print_char_limit} chars):")
            print(f"{'-'*80}")
            print(full_prompt_decoded[:print_char_limit])
            print(f"{'-'*80}\n")

        # Truncate if too long
        if max_len is not None and len(full_prompt) > max_len:
            full_prompt = self._truncate_sequence(
                segments=truncation_segments,
                truncatable_index=truncatable_index,
                max_len=max_len,
                keep_end=keep_end
            )

            # Debug: Print after truncation
            if debug:
                print(f"\n{'='*80}")
                print(f"DEBUG: {debug_label} Structure (AFTER truncation)")
                print(f"{'='*80}")
                print(f"Truncated prompt length: {len(full_prompt)} tokens (max: {max_len})")
                truncated_decoded = self.tokenizer.decode(full_prompt.long(), skip_special_tokens=True)
                print(f"\nTruncated prompt content (first {print_char_limit} chars):")
                print(f"{'-'*80}")
                print(truncated_decoded[:print_char_limit])
                print(f"{'-'*80}\n")

        return full_prompt

    def _replace_prompts_with_original(self, output: DataProto,
                                        original_prompt_ids_list: List[torch.Tensor]) -> None:
        """
        Replace extended prompts with original prompts for correct KL divergence computation.

        When generating with extended context (e.g., revision or transfer learning),
        the actual generation prompt includes additional context. However, for KL divergence
        computation during training, we need the prompts to match the original prompts.

        This method:
        1. Pads original prompts to the same length
        2. Replaces the 'prompts' field with original prompts
        3. Rebuilds 'input_ids' as original_prompts + responses
        4. Rebuilds 'attention_mask', 'info_mask', and 'position_ids' accordingly

        Args:
            output: DataProto to modify (modified in-place)
            original_prompt_ids_list: List of original prompt tensors
        """
        # Pad original prompts to same length
        original_prompts_padded_list = []
        max_original_prompt_len = max(len(p) for p in original_prompt_ids_list)
        for original_prompt_ids in original_prompt_ids_list:
            padded = torch.cat([
                torch.full((max_original_prompt_len - len(original_prompt_ids),),
                          self.tokenizer.pad_token_id,
                          dtype=original_prompt_ids.dtype,
                          device=original_prompt_ids.device),
                original_prompt_ids
            ])
            original_prompts_padded_list.append(padded)

        original_prompts_tensor = torch.stack(original_prompts_padded_list)

        # Replace prompts and rebuild tensors
        output.batch['prompts'] = original_prompts_tensor
        output.batch['input_ids'] = torch.cat([
            original_prompts_tensor,
            output.batch['responses']
        ], dim=1)

        output.batch['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(original_prompts_tensor),
            self.tensor_fn.create_attention_mask(output.batch['responses'])
        ], dim=1)

        output.batch['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(original_prompts_tensor),
            self.tensor_fn.create_attention_mask(output.batch['responses_with_info_mask'])
        ], dim=1)

        output.batch['position_ids'] = self.tensor_fn.create_position_ids(
            output.batch['attention_mask']
        )

    def _create_padded_batch(self, input_ids_list: List[torch.Tensor],
                              max_len: int = None) -> DataProto:
        """
        Pad a list of input_ids tensors to same length and create a DataProto batch.

        This is a common pattern used in revision and transfer learning where we need to:
        1. Pad variable-length input sequences to the same length
        2. Optionally truncate to max_len (keeping the end)
        3. Create attention mask and position ids
        4. Package into a DataProto batch

        Args:
            input_ids_list: List of 1D tensors containing input token ids
            max_len: Optional maximum length. If provided and sequences exceed this,
                     truncate from the beginning (keep the end)

        Returns:
            DataProto with 'input_ids', 'attention_mask', and 'position_ids'
        """
        # Pad to same length
        max_seq_len = max(len(ids) for ids in input_ids_list)
        input_ids = torch.stack([
            torch.cat([
                torch.full((max_seq_len - len(ids),), self.tokenizer.pad_token_id, dtype=ids.dtype),
                ids
            ]) for ids in input_ids_list
        ])

        # Truncate to max_len if specified (keep the end)
        if max_len is not None and input_ids.shape[1] > max_len:
            input_ids = input_ids[:, -max_len:]

        # Create attention mask and position ids
        attention_mask = self.tensor_fn.create_attention_mask(input_ids)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)

        # Create batch
        return DataProto.from_dict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        })

    def _filter_text_for_display(self, text: str) -> str:
        """
        Filter text to keep only English, Chinese characters, and common punctuation/symbols.
        Replaces other characters with a placeholder.

        Args:
            text: Input text to filter

        Returns:
            Filtered text with only allowed characters
        """
        def is_allowed_char(char):
            code_point = ord(char)
            # ASCII printable characters (includes English, digits, punctuation)
            if 32 <= code_point <= 126:
                return True
            # Chinese characters (CJK Unified Ideographs)
            if 0x4E00 <= code_point <= 0x9FFF:
                return True
            # Chinese punctuation
            if 0x3000 <= code_point <= 0x303F:
                return True
            # Newline, tab, carriage return
            if char in '\n\t\r':
                return True
            return False

        # Filter characters
        filtered_chars = []
        for char in text:
            if is_allowed_char(char):
                filtered_chars.append(char)
            else:
                # Replace non-allowed characters with a placeholder
                filtered_chars.append('�')  # Unicode replacement character

        # Collapse consecutive replacement characters
        result = ''.join(filtered_chars)
        while '��' in result:
            result = result.replace('��', '�')

        return result

    def print_readable_dataproto(self, data_dict: Dict, title: str = "Data", sample_indices: List[int] = None, truncate_lines: bool = False):
        """
        Print DataProto-like dictionary in readable string format.

        Args:
            data_dict: Dictionary containing tensor data (e.g., original_left_side, original_right_side)
            title: Title to display for this data
            sample_indices: List of sample indices to print. If None, prints all samples.
            truncate_lines: If True, limit output to first 50 lines. If False, print entire text.
        """
        # Add epoch, batch index, and revision round to title if available
        context_parts = []
        if self.current_epoch is not None:
            context_parts.append(f"Epoch {self.current_epoch}")
        if self.current_batch_idx is not None:
            context_parts.append(f"Batch {self.current_batch_idx}")
        if self.current_phase is not None:
            context_parts.append(f"Phase: {self.current_phase}")

        if context_parts:
            title_with_context = f"[{' | '.join(context_parts)}] {title}"
        else:
            title_with_context = title

        print("\n" + "="*80)
        print(f"{title_with_context}")
        print("="*80)

        # Check if we have both prompts and responses to print them together
        has_prompts = 'prompts' in data_dict and isinstance(data_dict['prompts'], torch.Tensor)
        has_responses = 'responses' in data_dict and isinstance(data_dict['responses'], torch.Tensor)

        if has_prompts and has_responses:
            # Print prompt-response pairs together for better readability
            prompts_tensor = data_dict['prompts']
            responses_tensor = data_dict['responses']

            print(f"\nPrompts Shape: {prompts_tensor.shape}")
            print(f"Responses Shape: {responses_tensor.shape}")
            print("-"*80)

            # Determine which samples to print
            if sample_indices is None:
                sample_indices_to_print = range(prompts_tensor.shape[0])
            else:
                sample_indices_to_print = sample_indices

            for i in sample_indices_to_print:
                if i >= prompts_tensor.shape[0]:
                    continue

                print(f"\n{'='*80}")
                print(f"SAMPLE {i}")
                print(f"{'='*80}")

                # Decode prompt
                prompt_to_decode = prompts_tensor[i].long() if prompts_tensor.dtype != torch.long else prompts_tensor[i]
                prompt_text = self.tokenizer.decode(prompt_to_decode, skip_special_tokens=True)
                prompt_token_count = (prompt_to_decode != self.tokenizer.pad_token_id).sum().item()
                prompt_filtered = self._filter_text_for_display(prompt_text)
                prompt_lines = prompt_filtered.split('\n')

                if truncate_lines and len(prompt_lines) > 25:
                    prompt_displayed = '\n'.join(prompt_lines[:25])
                    prompt_displayed += f"\n... (truncated, {len(prompt_lines) - 25} more lines)"
                else:
                    prompt_displayed = prompt_filtered

                print(f"{'-'*80}")
                print(f"\n[PROMPT] (Length: {prompt_token_count} tokens, {len(prompt_lines)} lines):")
                print(f"{prompt_displayed}")

                # Decode response
                response_to_decode = responses_tensor[i].long() if responses_tensor.dtype != torch.long else responses_tensor[i]
                response_text = self.tokenizer.decode(response_to_decode, skip_special_tokens=True)
                response_token_count = (response_to_decode != self.tokenizer.pad_token_id).sum().item()
                response_filtered = self._filter_text_for_display(response_text)
                response_lines = response_filtered.split('\n')

                if truncate_lines and len(response_lines) > 25:
                    response_displayed = '\n'.join(response_lines[:25])
                    response_displayed += f"\n... (truncated, {len(response_lines) - 25} more lines)"
                else:
                    response_displayed = response_filtered
                print(f"{'-'*80}")
                print(f"\n[RESPONSE] (Length: {response_token_count} tokens, {len(response_lines)} lines):")
                print(f"{response_displayed}")
                print("-"*80)

            # Print any other keys that aren't prompts or responses
            for key, tensor in data_dict.items():
                if key in ['prompts', 'responses'] or not isinstance(tensor, torch.Tensor):
                    continue
                print("\n" + "="*80)
                print(f"\n[{key}]")
                print(f"Shape: {tensor.shape}")
                print("-"*80)
                for i in sample_indices_to_print:
                    if i >= tensor.shape[0]:
                        continue
                    tensor_to_decode = tensor[i].long() if tensor.dtype != torch.long else tensor[i]
                    decoded_text = self.tokenizer.decode(tensor_to_decode, skip_special_tokens=True)
                    token_count = (tensor_to_decode != self.tokenizer.pad_token_id).sum().item()
                    filtered_text = self._filter_text_for_display(decoded_text)
                    lines = filtered_text.split('\n')
                    if truncate_lines and len(lines) > 50:
                        displayed_text = '\n'.join(lines[:50])
                        displayed_text += f"\n... (truncated, {len(lines) - 50} more lines)"
                    else:
                        displayed_text = filtered_text
                    print(f"\nSample {i} (Length: {token_count} tokens, {len(lines)} lines):")
                    print(f"{displayed_text}")
                    print("-"*80)
        else:
            # Original behavior: print each key separately
            for key, tensor in data_dict.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                print("\n" + "="*80)
                print(f"\n[{key}]")
                print(f"Shape: {tensor.shape}")
                print("-"*80)

                # Determine which samples to print
                if sample_indices is None:
                    sample_indices_to_print = range(tensor.shape[0])
                else:
                    sample_indices_to_print = sample_indices

                for i in sample_indices_to_print:
                    if i >= tensor.shape[0]:
                        continue

                    # Decode tensor to readable string
                    tensor_to_decode = tensor[i].long() if tensor.dtype != torch.long else tensor[i]
                    decoded_text = self.tokenizer.decode(tensor_to_decode, skip_special_tokens=True)
                    token_count = (tensor_to_decode != self.tokenizer.pad_token_id).sum().item()

                    # Filter text to remove non-English/Chinese characters
                    filtered_text = self._filter_text_for_display(decoded_text)

                    # Optionally limit to first 50 lines
                    lines = filtered_text.split('\n')
                    if truncate_lines and len(lines) > 50:
                        displayed_text = '\n'.join(lines[:50])
                        displayed_text += f"\n... (truncated, {len(lines) - 50} more lines)"
                    else:
                        displayed_text = filtered_text

                    print(f"\nSample {i} (Length: {token_count} tokens, {len(lines)} lines):")
                    print(f"{displayed_text}")
                    print("-"*80)

        print("="*80 + "\n")

    def _debug_print_batch(self, batch: DataProto, is_input: bool = True) -> DataProto:
        """Print debug batch if debug_llm_io is enabled. Returns batch unchanged."""
        if self.config.debug_llm_io:
            label = "INPUT" if is_input else "OUTPUT"
            key = 'input_ids' if is_input else 'responses'
            print("\n" + "="*80)
            print(f"[DEBUG LLM IO] {label} _generate_with_gpu_padding:")
            print(f"Phase: {self.current_phase}, Shape: {batch.batch[key].shape}")
            print("="*80)
            for i in range(min(4, batch.batch[key].shape[0])):  # Print first 4 samples
                decoded = self.tokenizer.decode(batch.batch[key][i].long(), skip_special_tokens=False)
                print(f"\n[{label} {i} - Length: {len(batch.batch[key][i])} tokens]:\n{decoded}")
            print("="*80 + "\n")
        return batch

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        # Ensure tensor is long/int type before decoding
        responses = responses.long() if responses.dtype != torch.long else responses
        responses_str = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        self._debug_print_batch(active_batch, is_input=True)

        num_gpus = self.config.num_gpus

        # Convert all tensors to fp32
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()

        if num_gpus <= 1:
            output = self.actor_rollout_wg.generate_sequences(active_batch)
            return self._debug_print_batch(output, is_input=False)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        if remainder == 0:
            output = self.actor_rollout_wg.generate_sequences(active_batch)
            return self._debug_print_batch(output, is_input=False)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Copy meta_info from original batch
        if hasattr(active_batch, 'meta_info') and active_batch.meta_info:
            padded_active_batch.meta_info.update(active_batch.meta_info)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return self._debug_print_batch(padded_output, is_input=False)

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        # Print original_left_side and original_right_side in readable format
        self.print_readable_dataproto(original_left_side, title="ORIGINAL_LEFT_SIDE (Initial Prompt)", sample_indices=[0,1,2,3])
        # self.print_readable_dataproto(original_right_side, title="ORIGINAL_RIGHT_SIDE (Initial Responses - Empty)")

        # here is the mask is at batch level, which prompt is active
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        # Example Scenario
        # For a batch of 3 samples:
        # Sample 0: Search → Search → Answer  (turns_stats[0] = 3)
        # Sample 1: Answer                    (turns_stats[1] = 1)
        # Sample 2: Search → Answer           (turns_stats[2] = 2)
        # The final turns_stats would be [3, 1, 2], representing that:
        # Sample 0 took 3 actions (2 searches + 1 answer)
        # Sample 1 took 1 action (immediate answer)
        # Sample 2 took 2 actions (1 search + 1 answer)
        # This metric helps you understand:
        # How much the agent is thinking/searching before answering
        # Distribution of action counts across the batch
        # Efficiency of the agent (fewer turns might be better or worse depending on task complexity)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })

            llm_start_time = time.time()
            gen_output = self._generate_with_gpu_padding(rollings_active)
            llm_elapsed = time.time() - llm_start_time
            print(f"[Step {step}] LLM generation time: {llm_elapsed:.2f}s")
            
            meta_info = gen_output.meta_info
            # Example
            # Input (raw LLM generation):
            # I need to search for information. <search>What is quantum computing</search> And then maybe I'll search again <search>Another query</search>
            # Output after _postprocess_responses():
            # I need to search for information. <search>What is quantum computing</search>
            # Everything after the first </search> tag is removed!
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            # The _example_level_pad() function at tensor_helper.py:50-75 restores the batch dimension to its original size by padding inactive samples.
            # The Problem It Solves
            # In the generation loop:
            # Only active samples (those that haven't finished yet) are passed to the LLM for generation
            # This creates a batch size mismatch between:
            # The original batch (e.g., size 8)
            # The active batch (e.g., size 5 if 3 samples already finished)
            # To maintain consistent batch dimensions throughout the pipeline, we need to pad back the responses from inactive samples
            # Example
            # Scenario:
            # Original batch size: 5
            # Active mask: [True, False, True, False, True]
            # Active samples: 3 (indices 0, 2, 4)
            # Input:
            # responses.shape = (3, 100)  # 3 active samples, 100 tokens each
            # responses_str = ["response_0", "response_2", "response_4"]
            # active_mask = [True, False, True, False, True]
            # Output:
            # padded_responses.shape = (5, 100)  # Back to original batch size
            # # padded_responses[0] = actual response from sample 0
            # # padded_responses[1] = all pad tokens
            # # padded_responses[2] = actual response from sample 2  
            # # padded_responses[3] = all pad tokens
            # # padded_responses[4] = actual response from sample 4

            # padded_responses_str = ["response_0", "", "response_2", "", "response_4"]
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            # Return Values Explained
            # 1. next_obs - List[str]
            # Next observations to feed back to the LLM
            # For search actions: Contains the search results wrapped in <information> tags
            # '\n\n<information>{search_result}</information>\n\n'
            # For answer actions: Empty string '' (episode ends, no more observations needed)
            # For invalid actions: Error message instructing how to format actions properly
            # For inactive samples: Empty string ''
            # Purpose: This is what the LLM sees in the next turn to continue its reasoning.
            # 2. dones - List[bool] (stored as List[int])
            # Episode termination flags
            # 1 (True): Episode is done for this sample
            # When action is 'answer' (agent provided final answer)
            # When sample is inactive
            # 0 (False): Episode continues
            # When action is 'search' (agent wants to search)
            # When action is invalid (agent needs to retry)
            # Purpose: Controls the agent loop - stops generating for samples that are done.
            # 3. valid_action - List[bool] (stored as List[int])
            # Whether the action was valid
            # 1 (True): Valid action
            # 'search' with proper formatting
            # 'answer' with proper formatting
            # 0 (False): Invalid action
            # No <search> or <answer> tags found
            # Malformed action
            # Sample is inactive
            # Purpose: Used for computing metrics like env/number_of_valid_action and env/ratio_of_valid_action (see ray_trainer.py:273-274).
            # 4. is_search - List[bool] (stored as List[int])
            # Whether the action was a search
            # 1 (True): Action is a search query
            # 0 (False): Action is answer, invalid, or inactive
            # Purpose: Used for computing search-specific metrics like env/number_of_valid_search and tracking search behavior (see ray_trainer.py:275-276).
            # 5. search_queries_per_sample - List[str]
            # The actual search query for each sample
            # For search actions: The query text (content between <search> tags)
            # "What is quantum computing"
            # For all other cases: Empty string ''
            # Purpose: Logging and debugging - tracks what queries the agent is making.
            # 6. search_results_per_sample - List[str]
            # The search results returned for each sample
            # For search actions: The full search result text from the search API
            # For all other cases: Empty string ''
            # Purpose: Logging and debugging - tracks what information the agent received.
            # Logic Flow Inside execute_predictions()
            # Step 1: Parse actions (line 830)
            # cur_actions, contents = self.postprocess_predictions(predictions)
            # Extracts action type ('search', 'answer', or None) and content from each prediction. Step 2: Batch search (lines 834-839)
            # Collects all search queries
            # Executes them in a single batch call to the search API
            # Gets results for all searches at once
            # Step 3: Process each sample (lines 845-879) Iterates through each sample and determines outputs based on:
            # Condition	next_obs	dones	valid_action	is_search	queries	results
            # Inactive	''	1	0	0	''	''
            # Answer	''	1	1	0	''	''
            # Search	<information>...</information>	0	1	1	query	result
            # Invalid	Error message	0	0	0	''	''
            # Example Scenario
            # Input:
            # predictions = [
            #     "Let me search for info. <search>quantum computing</search>",
            #     "I know this! <answer>42</answer>",
            #     "Invalid response without tags"
            # ]
            # active_mask = [True, True, True]
            # Output:
            # next_obs = [
            #     '\n\n<information>[search results for quantum computing]</information>\n\n',
            #     '',
            #     '\nMy previous action is invalid. If I want to search...'
            # ]
            # dones = [0, 1, 0]  # Continue, Done, Continue
            # valid_action = [1, 1, 0]  # Valid, Valid, Invalid
            # is_search = [1, 0, 0]  # Search, Not search, Not search
            # search_queries_per_sample = ['quantum computing', '', '']
            # search_results_per_sample = ['[search results]', '', '']


            search_start_time = time.time()
            next_obs, dones, valid_action, is_search, search_queries, search_results = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            search_elapsed = time.time() - search_start_time
            num_searches = sum(is_search)
            print(f"[Step {step}] Search engine time: {search_elapsed:.2f}s ({num_searches} searches)")

            # import pdb; pdb.set_trace()

            # Print search queries and results
            # print(f"\n{'='*60}")
            # print(f"STEP {step + 1} - Search Queries and Results")
            # print(f"{'='*60}")
            # for i, (query, result, is_search_flag) in enumerate(zip(search_queries, search_results, is_search)):
            #     if is_search_flag:
            #         print(f"\n[Sample {i}]")
            #         print(f"Query: {query}")
            #         print(f"Result (truncated to 200 chars):\n{result[:200]}...")
            #         print(f"{'-'*60}")
            # print(f"{'='*60}\n")

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            # Visual Example: Context Evolution
            # Let's trace how the context grows across multiple turns: Initial State (Turn 0):
            # rollings.batch['input_ids'] = [
            #     [PAD, PAD, "What", "is", "quantum", "computing", "?"]
            # ]
            # # Length: 7 tokens (5 real + 2 padding)
            # After Turn 1 (LLM searches): Input to _update_rolling_state():
            # rollings.batch['input_ids'] = [PAD, PAD, "What", "is", "quantum", "computing", "?"]
            # cur_responses = ["Let", "me", "search", "<search>", "quantum", "</search>"]
            # next_obs_ids = ["<information>", "Quantum", "computing", "uses", "qubits", "</information>"]
            # Step 1 - Concatenate:
            # new_input_ids = [
            #     PAD, PAD,
            #     "What", "is", "quantum", "computing", "?",           # Original prompt
            #     "Let", "me", "search", "<search>", "quantum", "</search>",  # LLM response
            #     "<information>", "Quantum", "computing", "uses", "qubits", "</information>"  # Observation
            # ]
            # # Length: 21 tokens
            # Step 2-3 - Create masks and position IDs:
            # attention_mask = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            # position_ids   = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            # Step 4-5 - Truncate if needed: If max_len = 4096 and effective_len = 19, we keep all tokens (no truncation). Output:
            # new_rollings = DataProto({
            #     'input_ids': [...all 21 tokens...],
            #     'attention_mask': [...],
            #     'position_ids': [...]
            # })
            # This becomes the input for Turn 2!
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            # _update_rolling_state()	Input to next LLM turn	Full conversation history (for generation)
            # _update_right_side()	Training data	Only responses + observations (for RL training)
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )

            # Print updated original_right_side after each step
            # self.print_readable_dataproto(
            #     original_right_side,
            #     title=f"ORIGINAL_RIGHT_SIDE (After Step {step})",
            #     sample_indices=[original_right_side['responses'].shape[0] - 1]  # Print only the last sample
            # )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })

            # Print rollings_active (input prompts) in readable string format
            # print("\n" + "="*80)
            # print("ROLLINGS_ACTIVE (FINAL ROLLOUT) - INPUT TO LLM (FULL CONTEXT):")
            # print(f"Shape: {rollings_active.batch['input_ids'].shape}")
            # print("="*80)
            # for i in range(rollings_active.batch['input_ids'].shape[0]):
            #     decoded_input = self.tokenizer.decode(rollings_active.batch['input_ids'][i], skip_special_tokens=False)
            #     print(f"\n[Input {i} - Length: {len(rollings_active.batch['input_ids'][i])} tokens]:\n{decoded_input}")
            # print("="*80 + "\n")

            llm_start_time = time.time()
            gen_output = self._generate_with_gpu_padding(rollings_active)
            llm_elapsed = time.time() - llm_start_time
            print(f"[Final rollout] LLM generation time: {llm_elapsed:.2f}s")

            # Print gen_output (generated responses) in readable string format
            # print("\n" + "="*80)
            # print("GEN_OUTPUT (FINAL ROLLOUT) - LLM GENERATED RESPONSES (NEW TOKENS ONLY):")
            # print(f"Shape: {gen_output.batch['responses'].shape}")
            # print("="*80)
            # for i in range(gen_output.batch['responses'].shape[0]):
            #     decoded_response = self.tokenizer.decode(gen_output.batch['responses'][i], skip_special_tokens=False)
            #     print(f"\n[Response {i} - Length: {len(gen_output.batch['responses'][i])} tokens]:\n{decoded_response}")
            # print("="*80 + "\n")

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search, search_queries, search_results = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            # Print search queries (final step, no actual search performed)
            # print(f"\n{'='*60}")
            # print(f"FINAL STEP - Answer Generation (No Search)")
            # print(f"{'='*60}\n")

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )

            # Print final original_right_side
            # self.print_readable_dataproto(
            #     original_right_side,
            #     title="ORIGINAL_RIGHT_SIDE (Final - After All Steps)",
            #     sample_indices=[original_right_side['responses'].shape[0] - 1]  # Print only the last sample
            # )

        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()

        # print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        # What it represents:
        # Mask that treats observation/information blocks as if they were padding
        # 1 = agent-generated token (LLM's own words)
        # 0 = padding OR environment-provided information (search results)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output

    def _extract_search_queries(self, response_text: str) -> List[str]:
        """
        Extract all search queries from a thinking process.

        Args:
            response_text: The decoded response text containing thinking process

        Returns:
            List of search queries found in the response
        """
        pattern = r'<search>(.*?)</search>'
        queries = re.findall(pattern, response_text, re.DOTALL)
        return [q.strip() for q in queries]

    def _perform_revision(self, gen_batch: DataProto,
                          initial_output: DataProto) -> DataProto:
        """
        Perform self-revision on the initial output.

        If two_step_revision is enabled:
            Step 1: Ask LLM to analyze the initial response
            Step 2: Add analysis to prompt and ask LLM to generate revised response
        Otherwise:
            Single step: Ask LLM to directly generate revised response

        Args:
            gen_batch: Original generation batch
            initial_output: Initial output to revise

        Returns:
            Revised DataProto output
        """
        

        if self.config.two_step_revision:
            # ==================== TWO-STEP REVISION ====================
            # Step 1: Generate analysis of the initial response
            self.current_phase = 'analysis'
            analysis_prompts, analysis_responses = self._generate_analysis(initial_output)

            # Step 2: Generate revised response based on analysis
            self.current_phase = 'revision'
            revision_output = self._generate_revision_with_analysis(
                gen_batch, initial_output, analysis_prompts, analysis_responses
            )
        else:
            # ==================== SINGLE-STEP REVISION (original behavior) ====================
            self.current_phase = 'revision'
            revision_output = self._generate_revision_single_step(gen_batch, initial_output)

        # Extract original prompts for KL divergence computation
        original_prompt_ids_list = [initial_output.batch['prompts'][i]
                                    for i in range(initial_output.batch['prompts'].shape[0])]

        # Replace extended prompts with original prompts for KL divergence computation
        self._replace_prompts_with_original(revision_output, original_prompt_ids_list)

        # Add revision metadata
        revision_output.meta_info['is_revision'] = True
        revision_output.meta_info['generation_type'] = 'revision'

        return revision_output

    def _generate_analysis(self, initial_output: DataProto) -> List[torch.Tensor]:
        """
        Step 1 of two-step revision: Generate analysis of the initial response.

        Args:
            initial_output: Initial output containing responses to analyze

        Returns:
            List of analysis response tensors for each sample
        """
        batch_size = initial_output.batch['responses'].shape[0]
        analysis_input_ids_list = []

        for i in range(batch_size):
            # Strip special tokens from original_prompt to avoid repeated <endoftext> tokens
            original_prompt_ids = self._strip_special_tokens(initial_output.batch['prompts'][i])
            previous_response_ids_full = self._strip_special_tokens(initial_output.batch['responses'][i])

            # Decode previous response, extract only <search> queries, then re-encode
            # This reduces context length by removing lengthy <think>, <information>, and <answer> sections
            previous_response_text = self.tokenizer.decode(previous_response_ids_full.long(), skip_special_tokens=False)
            filtered_response_text = self._extract_search_queries_only(previous_response_text)
            previous_response_ids = self.tokenizer.encode(
                filtered_response_text,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            # Encode instructions for analysis step (use specific analysis instruction)
            top_instruction_ids = self.tokenizer.encode(
                self.config.top_instruction_for_analysis,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            previous_response_instruction_ids = self.tokenizer.encode(
                self.config.previous_response_instruction,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            analysis_instruction_ids = self.tokenizer.encode(
                self.config.analysis_prompt,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            # Concatenate and truncate: original_prompt is also truncatable to ensure top_instruction_for_revision_with_analysis is preserved
            # Segments: [top_instruction_for_revision_with_analysis, original_prompt, previous_response_instruction, previous_response, analysis_instruction]
            # Priority: Keep top_instruction_for_revision_with_analysis and analysis_instruction, truncate original_prompt from beginning if needed
            full_analysis_input = self._concatenate_and_truncate(
                components=[
                    top_instruction_ids,
                    original_prompt_ids,
                    previous_response_instruction_ids,
                    previous_response_ids,
                    analysis_instruction_ids
                ],
                truncatable_index=1,  # Truncate original_prompt to preserve instructions
                max_len=self.config.max_start_length,
                keep_end=True  # Keep end of original_prompt (most recent part of question)
            )

            analysis_input_ids_list.append(full_analysis_input)

        # Create padded batch for analysis generation (truncated to max_start_length for consistency)
        analysis_batch = self._create_padded_batch(
            analysis_input_ids_list,
            max_len=self.config.max_start_length
        )

        # Generate analysis (single turn, no search needed)
        print(f"\n{'='*80}")
        print(f"STEP 1: Generating Analysis of Initial Response")
        print(f"{'='*80}\n")

        gen_output = self._generate_with_gpu_padding(analysis_batch)
        analysis_responses = gen_output.batch['responses']

        # Print analysis output for debugging
        self.print_readable_dataproto(
            {'prompts': analysis_batch.batch['input_ids'], 'responses': analysis_responses},
            title="Analysis Step Output",
            sample_indices=[0, 1, 2, 3]
        )

        return analysis_batch.batch['input_ids'], analysis_responses

    def _generate_revision_with_analysis(self, gen_batch: DataProto,
                                          initial_output: DataProto,
                                          analysis_prompts: torch.Tensor,
                                          analysis_responses: torch.Tensor) -> DataProto:
        """
        Step 2 of two-step revision: Generate revised response based on analysis.

        Args:
            gen_batch: Original generation batch
            initial_output: Initial output containing responses to revise
            analysis_prompts: Analysis prompts from step 1
            analysis_responses: Analysis responses from step 1

        Returns:
            Revised DataProto output
        """
        batch_size = initial_output.batch['responses'].shape[0]
        revision_input_ids_list = []

        for i in range(batch_size):
            # Strip special tokens from original_prompt to avoid repeated <endoftext> tokens
            original_prompt_ids = self._strip_special_tokens(initial_output.batch['prompts'][i])
            previous_response_ids_full = self._strip_special_tokens(initial_output.batch['responses'][i])
            analysis_ids = self._strip_special_tokens(analysis_responses[i])

            # Decode previous response, extract only <search> queries, then re-encode
            # This reduces context length by removing lengthy <think>, <information>, and <answer> sections
            previous_response_text = self.tokenizer.decode(previous_response_ids_full.long(), skip_special_tokens=False)
            filtered_response_text = self._extract_search_queries_only(previous_response_text)
            previous_response_ids = self.tokenizer.encode(
                filtered_response_text,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            # Encode instructions
            top_instruction_ids = self.tokenizer.encode(
                self.config.top_instruction_for_revision_with_analysis,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            previous_response_instruction_ids = self.tokenizer.encode(
                self.config.previous_response_instruction,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            # Use analysis_label instead of analysis_prompt since analysis is already generated
            analysis_label_ids = self.tokenizer.encode(
                self.config.analysis_label,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            revision_instruction_ids = self.tokenizer.encode(
                self.config.revision_instruction_for_two_step,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            # Concatenate and truncate: original_prompt is truncatable to preserve instructions
            # Segments: [top_instruction_for_revision_with_analysis, original_prompt, previous_response_instruction,
            #            previous_response, analysis_label, analysis, revision_instruction]
            # Priority: Keep top_instruction_for_revision_with_analysis and revision_instruction, truncate original_prompt if needed
            full_revision_input = self._concatenate_and_truncate(
                components=[
                    top_instruction_ids,
                    original_prompt_ids,
                    previous_response_instruction_ids,
                    previous_response_ids,
                    analysis_label_ids,
                    analysis_ids,
                    revision_instruction_ids
                ],
                truncatable_index=1,  # Truncate original_prompt to preserve instructions
                max_len=self.config.max_start_length,
                keep_end=True  # Keep end of original_prompt (most recent part)
            )

            revision_input_ids_list.append(full_revision_input)

        # Create padded batch for revision (truncated to max_start_length for consistency)
        revision_gen_batch = self._create_padded_batch(
            revision_input_ids_list,
            max_len=self.config.max_start_length
        )

        # Copy meta_info from original
        revision_gen_batch.meta_info.update(gen_batch.meta_info)

        # Run LLM loop for this revision
        print(f"\n{'='*80}")
        print(f"STEP 2: Generating Revised Response Based on Analysis")
        print(f"{'='*80}\n")

        revision_output = self.run_llm_loop(revision_gen_batch, revision_gen_batch.batch['input_ids'])

        # Print all information for each sample together
        sample_indices_to_print = [0, 1, 2, 3, 4, 5, 6]
        for i in sample_indices_to_print:
            if i >= revision_output.batch['prompts'].shape[0]:
                continue

            print(f"\n{'='*80}")
            print(f"Revision Output (2 Steps: Analysis and Revision) - SAMPLE {i}")
            print(f"{'='*80}")

            # 1. Analysis Prompt
            print(f"\n{'-'*80}")
            print(f"[1] Analysis Prompt:")
            print(f"{'-'*80}")
            analysis_prompt_ids = analysis_prompts[i].long()
            analysis_prompt_text = self.tokenizer.decode(analysis_prompt_ids, skip_special_tokens=True)
            print(analysis_prompt_text)

            # 2. Analysis Response
            print(f"\n{'-'*80}")
            print(f"[2] Analysis Response:")
            print(f"{'-'*80}")
            analysis_response_ids = analysis_responses[i].long()
            analysis_response_text = self.tokenizer.decode(analysis_response_ids, skip_special_tokens=True)
            print(analysis_response_text)

            # 3. Revision Prompt
            print(f"\n{'-'*80}")
            print(f"[3] Revision Prompt:")
            print(f"{'-'*80}")
            prompt_ids = revision_output.batch['prompts'][i].long()
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            print(prompt_text)

            # 4. Original Response
            print(f"\n{'-'*80}")
            print(f"[4] Original Response (Before Revision):")
            print(f"{'-'*80}")
            original_response_ids = initial_output.batch['responses'][i].long()
            original_response_text = self.tokenizer.decode(original_response_ids, skip_special_tokens=True)
            print(original_response_text)

            # 5. Revised Response
            print(f"\n{'-'*80}")
            print(f"[5] Revised Response (After Revision):")
            print(f"{'-'*80}")
            revised_response_ids = revision_output.batch['responses'][i].long()
            revised_response_text = self.tokenizer.decode(revised_response_ids, skip_special_tokens=True)
            print(revised_response_text)

            # 6. Standard Answer (Ground Truth) and Score
            print(f"\n{'-'*80}")
            print(f"[6] Standard Answer and Score for Revised Response:")
            print(f"{'-'*80}")
            try:
                if i < len(gen_batch):
                    data_item = gen_batch[i]
                    ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

                    # Import necessary functions
                    import re
                    from verl.utils.reward_score.qa_em import em_check

                    # Extract answer from revised response (get the last <answer> tag)
                    # Note: extract_solution requires 2+ <answer> tags (for revision scenario),
                    # but revised response may only have 1, so we extract manually
                    answer_pattern = r'<answer>(.*?)</answer>'
                    matches = list(re.finditer(answer_pattern, revised_response_text, re.DOTALL))

                    if matches:
                        # Get the last answer tag
                        extracted_answer = matches[-1].group(1).strip()
                    else:
                        extracted_answer = None

                    # Compute score
                    if extracted_answer is None:
                        score = 0
                    else:
                        score = 1 if em_check(extracted_answer, ground_truth['target']) else 0

                    # Print information (without printing the full solution string)
                    print(f"Golden answers: {ground_truth['target']}")
                    print(f"Extracted answer: {extracted_answer}")
                    print(f"EM Score: {score}")
                else:
                    print(f"(Not available - i={i}, len(gen_batch)={len(gen_batch)})")
            except (KeyError, IndexError, AttributeError) as e:
                print(f"(Not available - Error: {type(e).__name__}: {e})")
                # Debug: print available keys
                try:
                    data_item = gen_batch[i]
                    print(f"  non_tensor_batch keys: {data_item.non_tensor_batch.keys() if hasattr(data_item, 'non_tensor_batch') else 'no non_tensor_batch'}")
                    if hasattr(data_item, 'non_tensor_batch') and 'reward_model' in data_item.non_tensor_batch:
                        print(f"  reward_model keys: {data_item.non_tensor_batch['reward_model'].keys()}")
                except Exception as debug_e:
                    print(f"  Debug error: {debug_e}")

            print(f"\n{'='*80}\n")

        return revision_output

    def _generate_revision_single_step(self, gen_batch: DataProto,
                                        initial_output: DataProto) -> DataProto:
        """
        Original single-step revision: Directly generate revised response.

        Args:
            gen_batch: Original generation batch
            initial_output: Initial output containing responses to revise

        Returns:
            Revised DataProto output
        """
        batch_size = initial_output.batch['responses'].shape[0]
        revision_input_ids_list = []

        for i in range(batch_size):
            # Strip special tokens from original_prompt to avoid repeated <endoftext> tokens
            original_prompt_ids = self._strip_special_tokens(initial_output.batch['prompts'][i])
            previous_response_ids_full = self._strip_special_tokens(initial_output.batch['responses'][i])

            # Decode previous response, extract only <search> queries, then re-encode
            # This reduces context length by removing lengthy <think>, <information>, and <answer> sections
            previous_response_text = self.tokenizer.decode(previous_response_ids_full.long(), skip_special_tokens=False)
            filtered_response_text = self._extract_search_queries_only(previous_response_text)
            previous_response_ids = self.tokenizer.encode(
                filtered_response_text,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            # Encode instructions
            previous_response_instruction_ids = self.tokenizer.encode(
                self.config.previous_response_instruction,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            top_instruction_ids = self.tokenizer.encode(
                self.config.top_instruction_for_revision_single_step,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            revision_instruction_ids = self.tokenizer.encode(
                self.config.revision_instruction_for_single_step,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            # Concatenate and truncate: original_prompt is truncatable to preserve instructions
            # Segments: [top_instruction_for_revision_single_step, original_prompt, previous_response_instruction, previous_response, revision_instruction]
            # Priority: Keep top_instruction_for_revision_single_step and revision_instruction, truncate original_prompt if needed
            full_revision_input = self._concatenate_and_truncate(
                components=[
                    top_instruction_ids,
                    original_prompt_ids,
                    previous_response_instruction_ids,
                    previous_response_ids,
                    revision_instruction_ids
                ],
                truncatable_index=1,  # Truncate original_prompt to preserve instructions
                max_len=self.config.max_start_length,
                keep_end=True  # Keep end of original_prompt (most recent part)
            )

            revision_input_ids_list.append(full_revision_input)

        # Create padded batch for revision (truncated to max_start_length for consistency)
        revision_gen_batch = self._create_padded_batch(
            revision_input_ids_list,
            max_len=self.config.max_start_length
        )

        # Copy meta_info from original
        revision_gen_batch.meta_info.update(gen_batch.meta_info)

        # Run LLM loop for this revision
        revision_output = self.run_llm_loop(revision_gen_batch, revision_gen_batch.batch['input_ids'])

        # Print ACTUAL revision prompt that was used for generation
        self.print_readable_dataproto(
            {'prompts': revision_output.batch['prompts'], 'responses': revision_output.batch['responses']},
            title="Revision Output (Single-Step)",
            sample_indices=[0, 1, 2, 3]
        )

        return revision_output

    def _perform_transfer_learning(self, gen_batch: DataProto,
                                   current_output: DataProto,
                                   n_agent: int = 1) -> DataProto:
        """
        Perform transfer learning by creating prompts that include all previous thinking processes
        from the same original prompt (grouped by n_agent repetitions).

        The batch is structured as: [P0, P0, ..., P1, P1, ...] where each prompt is repeated n_agent times.
        For transfer learning, we group responses by their original prompt and use all of them.

        Args:
            gen_batch: Original generation batch containing the original prompts
            current_output: Current output containing all previous responses
            n_agent: Number of agents (repetitions) per original prompt

        Returns:
            New output from transfer learning generation (one response per n_agent group)
        """
        batch_size = current_output.batch['responses'].shape[0]
        num_unique_prompts = batch_size // n_agent

        print(f"\n{'='*80}")
        print("TRANSFER LEARNING - Creating prompts with all previous thinking processes")
        print(f"{'='*80}")
        print(f"Batch size: {batch_size}")
        print(f"n_agent: {n_agent}")
        print(f"Number of unique prompts: {num_unique_prompts}")
        print(f"Responses per prompt: {n_agent}")
        print()
        self.current_phase = 'transfer'

        # Create transfer learning prompts for each unique prompt (not per sample)
        transfer_input_ids_list = []
        original_prompt_ids_list = []  # Store original prompts to preserve for KL computation
        for prompt_idx in range(num_unique_prompts):
            # Calculate indices for this prompt group
            start_idx = prompt_idx * n_agent
            end_idx = start_idx + n_agent

            # Get original prompt (should be same for all in group)
            original_prompt_ids = current_output.batch['prompts'][start_idx]
            original_prompt_ids_list.append(original_prompt_ids)  # Save for later

            # Collect all prompts and responses for this prompt group
            all_prompts_for_group = []
            all_responses_for_prompt = []
            for agent_idx in range(start_idx, end_idx):
                prompt_ids = current_output.batch['prompts'][agent_idx]
                response_ids = current_output.batch['responses'][agent_idx]
                all_prompts_for_group.append(prompt_ids)
                all_responses_for_prompt.append(response_ids)

            # # Print debug info for ALL prompts (not just first)
            # print(f"\n{'='*80}")
            # print(f"PROMPT-RESPONSE MATCHING VERIFICATION (Prompt Group {prompt_idx})")
            # print(f"{'='*80}\n")

            # # Print all prompts in this group to verify they are identical
            # print(f"ALL PROMPTS IN THIS GROUP ({len(all_prompts_for_group)} prompts, should be identical):")
            # print(f"{'-'*80}\n")
            # for i, prompt in enumerate(all_prompts_for_group):
            #     prompt_decoded = self.tokenizer.decode(prompt, skip_special_tokens=True)
            #     prompt_filtered = self._filter_text_for_display(prompt_decoded)
            #     prompt_lines = prompt_filtered.split('\n')

            #     print(f"Prompt {i} ({(prompt != self.tokenizer.pad_token_id).sum().item()} tokens, {len(prompt_lines)} lines):")
            #     print(f"{'-'*80}")
            #     print(prompt_filtered)  # Print entire text without truncation
            #     print(f"\n{'-'*80}\n")

            # print(f"{'='*80}\n")

            # # Print all responses
            # print(f"ALL RESPONSES FOR THIS PROMPT GROUP ({len(all_responses_for_prompt)} responses):")
            # print(f"{'-'*80}\n")
            # for i, resp in enumerate(all_responses_for_prompt):
            #     resp_decoded = self.tokenizer.decode(resp, skip_special_tokens=True)
            #     resp_filtered = self._filter_text_for_display(resp_decoded)
            #     resp_lines = resp_filtered.split('\n')

            #     print(f"Response {i} ({(resp != self.tokenizer.pad_token_id).sum().item()} tokens, {len(resp_lines)} lines):")
            #     print(resp_filtered)  # Print entire text without truncation
            #     print(f"\n{'-'*80}\n")

            # print(f"{'='*80}\n")

            # if prompt_idx == 0:  # Only pause at first prompt group for interactive debugging
            #     import pdb; pdb.set_trace()

            # Create transfer learning prompt using ALL responses from this prompt
            transfer_prompt_ids = self._create_transfer_learning_prompt(
                original_prompt_ids,
                all_responses_for_prompt
            )

            # if prompt_idx == 0:
            #     print(f"  - Transfer prompt length: {len(transfer_prompt_ids)} tokens")
            #     print()

            transfer_input_ids_list.append(transfer_prompt_ids)

        # Create padded batch for transfer learning (truncated to max_start_length for consistency)
        transfer_gen_batch = self._create_padded_batch(
            transfer_input_ids_list,
            max_len=self.config.max_start_length
        )

        # Copy meta_info from original
        transfer_gen_batch.meta_info.update(gen_batch.meta_info)

        # Note: UIDs are not available at generation time - they're added later in ray_trainer.py
        # The UID handling happens after generation when batches are merged in the training pipeline

        # Run LLM loop for transfer learning
        print(f"\n{'='*80}")
        print("TRANSFER LEARNING - Generating new thinking process")
        print(f"{'='*80}\n")

        transfer_output = self.run_llm_loop(transfer_gen_batch, transfer_gen_batch.batch['input_ids'])

        # Print ACTUAL transfer learning prompt that was used for generation (before replacement)
        self.print_readable_dataproto(
            {'prompts': transfer_output.batch['prompts'], 'responses': transfer_output.batch['responses']},
            title=f"Transfer Learning Output (ACTUAL prompts used for generation - includes all previous responses + instruction)",
            sample_indices=[0,1,2,3,4,5]  # Print first sample
        )

        # Replace extended prompts with original prompts for correct KL divergence computation
        self._replace_prompts_with_original(transfer_output, original_prompt_ids_list)

        # Add metadata
        transfer_output.meta_info['is_transfer_learning'] = True

        # # Print transfer learning output in readable format
        # self.print_readable_dataproto(
        #     {'prompts': transfer_output.batch['prompts'], 'responses': transfer_output.batch['responses']},
        #     title=f"Transfer Learning Output (with original prompt restored)",
        #     sample_indices=[0]  # Print first sample
        # )

        return transfer_output

    def _create_transfer_learning_prompt(self, original_prompt_ids: torch.Tensor,
                                        all_responses: List[torch.Tensor]) -> torch.Tensor:
        """
        Create a transfer learning prompt that includes the original prompt and all previous thinking processes.

        Args:
            original_prompt_ids: The original prompt token IDs
            all_responses: List of all previous response token IDs for this prompt

        Returns:
            Combined prompt for transfer learning
        """
        # Strip special tokens from original prompt to avoid repeated <endoftext> tokens
        original_prompt_ids_clean = self._strip_special_tokens(original_prompt_ids)

        # Encode top instruction for transfer learning
        top_instruction_ids = self.tokenizer.encode(
            self.config.top_instruction_for_transfer,
            add_special_tokens=False,
            return_tensors='pt'
        ).squeeze(0)

        # Start with top instruction and original prompt
        components = [top_instruction_ids, original_prompt_ids_clean]

        # Add all previous thinking processes with instruction before each
        # Extract only search queries to reduce context length
        for idx, response_ids in enumerate(all_responses, 1):
            # Decode response, extract only <search> queries, then re-encode
            # This reduces context length by removing lengthy <think>, <information>, and <answer> sections
            response_text = self.tokenizer.decode(response_ids.long(), skip_special_tokens=False)
            filtered_response_text = self._extract_search_queries_only(response_text)
            filtered_response_ids = self.tokenizer.encode(
                filtered_response_text,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)

            # Add instruction before each response (now showing search queries only)
            response_instruction_ids = self.tokenizer.encode(
                f"\n\nPrevious thinking process {idx} (search queries only):\n",
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)
            components.append(response_instruction_ids)
            components.append(filtered_response_ids)

        # Add transfer learning instruction
        transfer_instruction_ids = self.tokenizer.encode(
            self.config.transfer_prompt,
            add_special_tokens=False,
            return_tensors='pt'
        ).squeeze(0)
        components.append(transfer_instruction_ids)

        # Prepare truncation segments: [top_instruction, original_prompt, all_responses_concat, transfer_instruction]
        response_components = components[2:-1]  # All response instructions + responses (skip top_instruction and original_prompt)
        all_responses_concat = torch.cat(response_components, dim=0) if response_components else torch.tensor([], dtype=original_prompt_ids_clean.dtype)

        # Use helper to concatenate, debug print, and truncate
        # Segments: [top_instruction_for_transfer, original_prompt, all_responses_concat, transfer_instruction]
        # Priority: Keep top_instruction_for_transfer and transfer_instruction, truncate original_prompt if needed
        full_prompt = self._concatenate_and_truncate(
            components=components,
            truncation_segments=[top_instruction_ids, original_prompt_ids_clean, all_responses_concat, transfer_instruction_ids],
            truncatable_index=1,  # Truncate original_prompt (index 1) to preserve top_instruction and transfer_instruction
            max_len=self.config.max_start_length,
            keep_end=True,  # Keep end of original_prompt (most recent part)
            debug=True,
            debug_label="Transfer Learning Prompt",
            debug_info={
                "Top instruction length": f"{len(top_instruction_ids)} tokens",
                "Original prompt length": f"{len(original_prompt_ids_clean)} tokens",
                "Number of responses": len(all_responses),
                "Transfer instruction length": f"{len(transfer_instruction_ids)} tokens"
            },
            print_char_limit=6000
        )

        return full_prompt

    def _merge_revision_and_transfer_outputs(self, revision_output: DataProto,
                                            transfer_output: DataProto,
                                            num_unique_prompts: int) -> DataProto:
        """
        Merge revision and transfer learning outputs with group-level concatenation.

        Structure:
        - revision_output: N × n_agent samples (e.g., [R0_a0, R0_a1, R1_a0, R1_a1])
        - transfer_output: N samples (e.g., [T0, T1])
        - Result: N × (n_agent + 1) samples grouped by prompt
                 (e.g., [R0_a0, R0_a1, T0, R1_a0, R1_a1, T1])

        Args:
            revision_output: Output from revision containing N × n_agent samples
            transfer_output: Output from transfer learning containing N samples
            num_unique_prompts: Number of unique prompts (N)

        Returns:
            Merged DataProto with group-level concatenation
        """
        print(f"\n{'='*80}")
        print(f"MERGING REVISION AND TRANSFER LEARNING OUTPUTS")
        print(f"{'='*80}\n")

        revision_batch_size = revision_output.batch['responses'].shape[0]
        transfer_batch_size = transfer_output.batch['responses'].shape[0]

        print(f"Revision batch size: {revision_batch_size}")
        print(f"Transfer batch size: {transfer_batch_size}")
        print(f"n_agent: {self.n_agent}")
        print(f"Num unique prompts: {num_unique_prompts}")

        # Concatenate revision and transfer at group level
        # revision: [R0_a0, R0_a1, R1_a0, R1_a1] (N*n_agent samples)
        # transfer: [T0, T1] (N samples)
        # Result: [R0_a0, R0_a1, T0, R1_a0, R1_a1, T1] (N*n_agent + N samples)
        # Groups: uid_0=[R0_a0, R0_a1, T0], uid_1=[R1_a0, R1_a1, T1]

        concatenated_batch = {}
        for key in revision_output.batch.keys():
            rev = revision_output.batch[key]  # Shape: (N*n_agent, seq_len_rev)
            trans = transfer_output.batch[key]  # Shape: (N, seq_len_trans)

            # Determine the max length for THIS specific key
            if len(rev.shape) > 1 and len(trans.shape) > 1:
                max_seq_len_for_key = max(rev.shape[1], trans.shape[1])
            else:
                # For 1D tensors, no padding needed
                group_list = []
                for prompt_idx in range(num_unique_prompts):
                    start_idx = prompt_idx * self.n_agent
                    end_idx = start_idx + self.n_agent
                    group_list.append(rev[start_idx:end_idx])
                    group_list.append(trans[prompt_idx:prompt_idx+1])
                concatenated_batch[key] = torch.cat(group_list, dim=0)
                continue

            # Determine the appropriate padding value based on the key
            if key in ['attention_mask', 'info_mask']:
                # Masks should be padded with 0 (don't attend)
                pad_value = 0
            elif key == 'position_ids':
                # Position IDs should be padded with 0 (will be ignored due to attention mask)
                pad_value = 0
            else:
                # Token IDs (prompts, responses, input_ids) should be padded with pad_token_id
                pad_value = self.tokenizer.pad_token_id

            # Pad both revision and transfer to the same max length FOR THIS KEY
            if rev.shape[1] < max_seq_len_for_key:
                padding = torch.full((rev.shape[0], max_seq_len_for_key - rev.shape[1]),
                                    pad_value,
                                    dtype=rev.dtype,
                                    device=rev.device)
                rev = torch.cat([padding, rev], dim=1)  # Left padding
                print(f"Padded revision {key} from {revision_output.batch[key].shape[1]} to {max_seq_len_for_key}")

            if trans.shape[1] < max_seq_len_for_key:
                padding = torch.full((trans.shape[0], max_seq_len_for_key - trans.shape[1]),
                                    pad_value,
                                    dtype=trans.dtype,
                                    device=trans.device)
                trans = torch.cat([padding, trans], dim=1)  # Left padding
                print(f"Padded transfer {key} from {transfer_output.batch[key].shape[1]} to {max_seq_len_for_key}")

            # Interleave at group level:
            # For each prompt group, concatenate n_agent revision responses + 1 transfer response
            group_list = []
            for prompt_idx in range(num_unique_prompts):
                start_idx = prompt_idx * self.n_agent
                end_idx = start_idx + self.n_agent
                # Add all revision responses for this prompt
                group_list.append(rev[start_idx:end_idx])
                # Add the transfer response for this prompt
                group_list.append(trans[prompt_idx:prompt_idx+1])

            concatenated_batch[key] = torch.cat(group_list, dim=0)

        # Handle non_tensor_batches similarly
        concatenated_non_tensor_batch = {}
        if hasattr(revision_output, 'non_tensor_batch') and revision_output.non_tensor_batch:
            for key in revision_output.non_tensor_batch.keys():
                rev_nt = revision_output.non_tensor_batch[key]
                trans_nt = transfer_output.non_tensor_batch[key]

                # Interleave at group level
                group_list = []
                for prompt_idx in range(num_unique_prompts):
                    start_idx = prompt_idx * self.n_agent
                    end_idx = start_idx + self.n_agent
                    # Add all revision responses for this prompt
                    group_list.append(rev_nt[start_idx:end_idx])
                    # Add the transfer response for this prompt
                    group_list.append(trans_nt[prompt_idx:prompt_idx+1])

                concatenated_non_tensor_batch[key] = np.concatenate(group_list, axis=0)

        # Create merged output with concatenated data
        merged_output = DataProto.from_dict(concatenated_batch)
        if concatenated_non_tensor_batch:
            merged_output.non_tensor_batch = concatenated_non_tensor_batch

        # Copy meta_info from revision output
        merged_output.meta_info.update(revision_output.meta_info)

        # Update meta_info to indicate this is a merged output
        merged_output.meta_info['is_merged_output'] = True
        merged_output.meta_info['revision_batch_size'] = revision_batch_size
        merged_output.meta_info['transfer_batch_size'] = transfer_batch_size
        merged_output.meta_info['total_batch_size'] = revision_batch_size + transfer_batch_size

        print(f"Merged output batch size: {revision_batch_size} (revision) + {transfer_batch_size} (transfer) = {revision_batch_size + transfer_batch_size} (total)")
        print(f"Structure: Each prompt group has {self.n_agent} revision responses + 1 transfer response = {self.n_agent + 1} responses per group")
        print(f"\n{'='*80}")
        print(f"REVISION AND TRANSFER LEARNING MERGED SUCCESSFULLY")
        print(f"{'='*80}\n")

        return merged_output

    def run_llm_loop_self_evolve(self, gen_batch, initial_input_ids: torch.Tensor) -> DataProto:
        """
        Wrapper around run_llm_loop that adds self-evolution capability (revision + transfer learning).

        After the initial generation, the LLM can review its own work and revise it
        for a configurable number of rounds, and optionally apply transfer learning
        to learn from other responses to the same prompt.

        Args:
            gen_batch: Initial generation batch
            initial_input_ids: Initial input IDs

        Returns:
            DataProto with final evolved output
        """
        # Skip revision and transfer during validation to maintain original batch size
        is_validation = gen_batch.meta_info.get('validate', False)
        if is_validation:
            print(f"VALIDATION MODE - Skipping revision and transfer learning")
            return self.run_llm_loop(gen_batch, initial_input_ids)

        print(f"\n{'='*80}")
        print(f"run_llm_loop_self_evolve (revision={self.config.enable_revision}, transfer_learning={self.config.enable_transfer_learning})")
        print(f"{'='*80}\n")

        # Run initial generation
        print(f"\n{'='*80}")
        print("INITIAL GENERATION")
        print(f"{'='*80}\n")
        self.current_phase = 'initial'
        current_output = self.run_llm_loop(gen_batch, initial_input_ids)

        # Print initial generation output in readable format
        self.print_readable_dataproto(
            {'prompts': current_output.batch['prompts'], 'responses': current_output.batch['responses']},
            title="Initial Generation Output",
            sample_indices=[0,1,2,3]  # Print first sample
        )
        print(f"\n{'='*80}")
        print("INITIAL GENERATION COMPLETED")
        print(f"{'='*80}\n")

        # Store original output for transfer learning if needed
        original_output = current_output

        # Self-revision (single round only)
        if self.config.enable_revision:
            print(f"\n{'='*80}")
            print(f"SELF-REVISION")
            print(f"{'='*80}\n")
            current_output = self._perform_revision(gen_batch, current_output)
            print(f"\n{'='*80}")
            print(f"SELF-REVISION COMPLETED")
            print(f"{'='*80}\n")

        # After revision (or if no revision), apply transfer learning if enabled
        if self.config.enable_transfer_learning:
            # Choose which responses to use for transfer learning based on config
            if self.config.transfer_use_revised and self.config.enable_revision:
                # Use revised responses for transfer learning
                print(f"Using REVISED responses for transfer learning")
                transfer_input = current_output
            else:
                # Use original responses for transfer learning
                print(f"Using ORIGINAL responses for transfer learning")
                transfer_input = original_output

            # Perform transfer learning
            print(f"\n{'='*80}")
            print(f"TRANSFER LEARNING")
            print(f"{'='*80}\n")

            transfer_output = self._perform_transfer_learning(gen_batch, transfer_input, self.n_agent)
            transfer_output.meta_info['is_transfer'] = True
            transfer_output.meta_info['generation_type'] = 'transfer_learning'

            # Merge base and transfer learning outputs
            transfer_batch_size = transfer_output.batch['responses'].shape[0]
            current_output = self._merge_revision_and_transfer_outputs(
                revision_output=transfer_input,
                transfer_output=transfer_output,
                num_unique_prompts=transfer_batch_size
            )

            print(f"\n{'='*80}")
            print(f"TRANSFER LEARNING COMPLETED AND MERGED")
            print(f"{'='*80}\n")

        print(f"\n{'='*80}")
        print(f"run_llm_loop_self_evolve COMPLETE")
        print(f"{'='*80}\n")

        # import pdb; pdb.set_trace()
        return current_output

        # # VERIFICATION: Check that all keys in current_output are correct and consistent
        # print(f"\n{'='*80}")
        # print("VERIFICATION - Final current_output after revision and transfer")
        # print(f"{'='*80}")

        # # 1. Check all keys and their shapes
        # print("\n1. Keys and Shapes:")
        # for key, value in current_output.batch.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"   {key}: {value.shape} (dtype: {value.dtype})")

        # # 2. Verify batch size consistency
        # print("\n2. Batch Size Consistency:")
        # batch_sizes = {key: value.shape[0] for key, value in current_output.batch.items() if isinstance(value, torch.Tensor)}
        # unique_batch_sizes = set(batch_sizes.values())
        # if len(unique_batch_sizes) == 1:
        #     print(f"   ✓ All tensors have same batch size: {list(unique_batch_sizes)[0]}")
        # else:
        #     raise ValueError(f"Batch size inconsistency detected! Different tensors have different batch sizes: {batch_sizes}")

        # # 3. Verify sequence length consistency for related keys
        # print("\n3. Sequence Length Consistency:")
        # if 'prompts' in current_output.batch and 'responses' in current_output.batch and 'input_ids' in current_output.batch:
        #     prompt_len = current_output.batch['prompts'].shape[1]
        #     response_len = current_output.batch['responses'].shape[1]
        #     input_len = current_output.batch['input_ids'].shape[1]
        #     print(f"   prompts length: {prompt_len}")
        #     print(f"   responses length: {response_len}")
        #     print(f"   input_ids length: {input_len}")
        #     if input_len == prompt_len + response_len:
        #         print(f"   ✓ input_ids = prompts + responses ({input_len} = {prompt_len} + {response_len})")
        #     else:
        #         raise ValueError(f"Sequence length mismatch! input_ids ({input_len}) != prompts ({prompt_len}) + responses ({response_len})")

        # # 4. Verify responses and responses_with_info_mask have same length
        # print("\n4. Response Mask Consistency:")
        # if 'responses' in current_output.batch and 'responses_with_info_mask' in current_output.batch:
        #     resp_len = current_output.batch['responses'].shape[1]
        #     resp_mask_len = current_output.batch['responses_with_info_mask'].shape[1]
        #     print(f"   responses length: {resp_len}")
        #     print(f"   responses_with_info_mask length: {resp_mask_len}")
        #     if resp_len == resp_mask_len:
        #         print(f"   ✓ Same length: {resp_len}")
        #     else:
        #         raise ValueError(f"Response mask length mismatch! responses ({resp_len}) != responses_with_info_mask ({resp_mask_len})")

        # # 5. Verify attention_mask and input_ids have same length
        # print("\n5. Attention Mask Consistency:")
        # if 'attention_mask' in current_output.batch and 'input_ids' in current_output.batch:
        #     attn_len = current_output.batch['attention_mask'].shape[1]
        #     input_len = current_output.batch['input_ids'].shape[1]
        #     print(f"   attention_mask length: {attn_len}")
        #     print(f"   input_ids length: {input_len}")
        #     if attn_len == input_len:
        #         print(f"   ✓ Same length: {attn_len}")
        #     else:
        #         raise ValueError(f"Attention mask length mismatch! attention_mask ({attn_len}) != input_ids ({input_len})")

        # # 6. Verify info_mask and input_ids have same length
        # print("\n6. Info Mask Consistency:")
        # if 'info_mask' in current_output.batch and 'input_ids' in current_output.batch:
        #     info_len = current_output.batch['info_mask'].shape[1]
        #     input_len = current_output.batch['input_ids'].shape[1]
        #     print(f"   info_mask length: {info_len}")
        #     print(f"   input_ids length: {input_len}")
        #     if info_len == input_len:
        #         print(f"   ✓ Same length: {info_len}")
        #     else:
        #         raise ValueError(f"Info mask length mismatch! info_mask ({info_len}) != input_ids ({input_len})")

        # # 7. Verify position_ids and input_ids have same length
        # print("\n7. Position IDs Consistency:")
        # if 'position_ids' in current_output.batch and 'input_ids' in current_output.batch:
        #     pos_len = current_output.batch['position_ids'].shape[1]
        #     input_len = current_output.batch['input_ids'].shape[1]
        #     print(f"   position_ids length: {pos_len}")
        #     print(f"   input_ids length: {input_len}")
        #     if pos_len == input_len:
        #         print(f"   ✓ Same length: {pos_len}")
        #     else:
        #         raise ValueError(f"Position IDs length mismatch! position_ids ({pos_len}) != input_ids ({input_len})")

        # # 8. Verify responses_with_info_mask has pad tokens where info_mask is 0
        # print("\n8. Info Mask Correctness:")
        # if 'responses_with_info_mask' in current_output.batch and 'info_mask' in current_output.batch:
        #     responses_with_mask = current_output.batch['responses_with_info_mask']
        #     info_mask = current_output.batch['info_mask']

        #     # Get the response portion of info_mask (last response_len tokens)
        #     response_len = responses_with_mask.shape[1]
        #     response_info_mask = info_mask[:, -response_len:]

        #     # Where info_mask is 0, responses_with_info_mask should be pad_token_id
        #     masked_positions = (response_info_mask == 0)
        #     if masked_positions.any():
        #         values_at_masked_positions = responses_with_mask[masked_positions]
        #         all_pad_tokens = (values_at_masked_positions == self.tokenizer.pad_token_id).all()

        #         num_masked = masked_positions.sum().item()
        #         num_correct_pads = (values_at_masked_positions == self.tokenizer.pad_token_id).sum().item()

        #         print(f"   Masked positions (info_mask=0): {num_masked}")
        #         print(f"   Correct pad tokens at masked positions: {num_correct_pads}/{num_masked}")

        #         if all_pad_tokens:
        #             print(f"   ✓ All masked positions have pad_token_id")
        #         else:
        #             num_incorrect = num_masked - num_correct_pads
        #             # Sample some incorrect positions for debugging
        #             incorrect_mask = masked_positions & (responses_with_mask != self.tokenizer.pad_token_id)
        #             sample_incorrect_values = responses_with_mask[incorrect_mask][:5]  # Show first 5
        #             raise ValueError(f"Info mask verification failed! {num_incorrect}/{num_masked} masked positions do not have pad_token_id. Sample incorrect values: {sample_incorrect_values.tolist()}")
        #     else:
        #         print(f"   ✓ No masked positions (all info_mask=1)")

        # # 9. Verify prompts are the ORIGINAL prompts (not extended)
        # print("\n9. Original Prompt Verification (sampling first prompt):")
        # if 'prompts' in current_output.batch:
        #     first_prompt = current_output.batch['prompts'][0]
        #     first_prompt_decoded = self.tokenizer.decode(first_prompt, skip_special_tokens=True)
        #     # Check if it contains revision instruction (it shouldn't!)
        #     has_revision_instr = self.config.revision_instruction_for_single_step[:50] in first_prompt_decoded if hasattr(self.config, 'revision_instruction_for_single_step') else False
        #     has_transfer_instr = "thinking processes" in first_prompt_decoded.lower()

        #     print(f"   First prompt length: {(first_prompt != self.tokenizer.pad_token_id).sum().item()} tokens")
        #     print(f"   Contains revision instruction: {has_revision_instr}")
        #     print(f"   Contains transfer instruction: {has_transfer_instr}")

        #     if not has_revision_instr and not has_transfer_instr:
        #         print(f"   ✓ Prompts appear to be ORIGINAL (no revision/transfer instructions)")
        #     else:
        #         error_msg = f"Prompt verification failed! Prompts contain extended context (revision_instr={has_revision_instr}, transfer_instr={has_transfer_instr}). First 500 chars: {first_prompt_decoded[:500]}"
        #         raise ValueError(error_msg)

        # print(f"\n{'='*80}\n")


    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> Tuple[List[str], List[bool], List[bool], List[bool], List[str], List[str]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM

        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding

        Returns:
            Tuple of (next_obs, dones, valid_action, is_search, search_queries_per_sample, search_results_per_sample)
            - next_obs: List of observation strings
            - dones: List of done flags
            - valid_action: List of valid action flags
            - is_search: List of search flags
            - search_queries_per_sample: List of search queries for each sample (empty string if not search)
            - search_results_per_sample: List of search results for each sample (empty string if not search)
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        search_queries_per_sample, search_results_per_sample = [], []

        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        # Create a copy of search_results for logging
        search_results_copy = search_results.copy()
        search_query_idx = 0

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):

            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                search_queries_per_sample.append('')
                search_results_per_sample.append('')
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                    search_queries_per_sample.append('')
                    search_results_per_sample.append('')
                elif action == 'search':
                    result = search_results.pop(0)
                    next_obs.append(f'\n\n<information>{result.strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    search_queries_per_sample.append(search_queries[search_query_idx])
                    search_results_per_sample.append(result)
                    search_query_idx += 1
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    search_queries_per_sample.append('')
                    search_results_per_sample.append('')

        assert len(search_results) == 0

        return next_obs, dones, valid_action, is_search, search_queries_per_sample, search_results_per_sample

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
