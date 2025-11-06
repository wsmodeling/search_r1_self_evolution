# Reflection Agent

The Reflection Agent is designed to analyze poor LLM outputs and generate reflections to improve future performance by summarizing shortcomings and lessons learned.

## Overview

The reflection agent takes:
- An input image
- The correct answer for a task
- Multiple negative/poor LLM outputs
- Optional task context

And generates a comprehensive reflection paragraph that identifies:
1. Common patterns of errors
2. Specific shortcomings in reasoning or approach  
3. Key lessons to avoid similar mistakes in the future

## Key Features

- **Multi-modal Support**: Works with both text and images
- **Batch Processing**: Can handle multiple samples at once
- **Model Agnostic**: Compatible with different VLM architectures (Qwen, LLaMA, Phi3V, etc.)
- **Structured Output**: Generates coherent reflection paragraphs
- **Export Functionality**: Save results to JSON files

## Usage

### Single Sample Reflection

```python
from reflection_agent import create_reflection_agent

# Create the agent
agent = create_reflection_agent("/path/to/your/model")

# Generate reflection for a single sample
reflection = agent.generate_reflection(
    image_path="/path/to/image.jpg",
    correct_answer="change_color(obj1, red); change_position(obj2, left)",
    negative_samples=[
        "change_size(obj1, large)",
        "change_color(obj1, blue); change_position(obj2, right)",
        "change_material(obj1, metal)"
    ],
    task_context="spatial transformation"
)

print(reflection)
```

### Batch Processing

```python
# Prepare batch data
batch_data = [
    {
        "image_path": "/path/to/image1.jpg",
        "correct_answer": "answer1",
        "negative_samples": ["sample1", "sample2"],
        "task_context": "spatial reasoning"
    },
    {
        "image_path": "/path/to/image2.jpg", 
        "correct_answer": "answer2",
        "negative_samples": ["sample3", "sample4"],
        "task_context": "math reasoning"
    }
]

# Generate batch reflections
reflections = agent.generate_batch_reflections(batch_data)

# Save results
results = []
for i, reflection in enumerate(reflections):
    results.append({
        **batch_data[i],
        "reflection": reflection
    })

agent.save_reflection_results(results, "reflections_output.json")
```

### Command Line Usage

```bash
python reflection_agent.py \
    --model_name_or_path /path/to/model \
    --image_path /path/to/image.jpg \
    --correct_answer "correct answer here" \
    --negative_samples "sample1" "sample2" "sample3" \
    --task_context "spatial transformation" \
    --output_file "reflection_results.json"
```

## Prompt Template

The agent uses the following prompt structure:

```
Given the negative samples for {task_context} below, summarize the shortcomings of these negative samples and the lessons learned on how to avoid generating similar ones into a single paragraph. Translate this entire passage into English.

Correct Answer: {correct_answer}

Negative Samples:
Sample 1: {sample1}
Sample 2: {sample2}
...

Please analyze these negative samples and provide a comprehensive reflection that identifies:
1. Common patterns of errors
2. Specific shortcomings in reasoning or approach
3. Key lessons to avoid similar mistakes in the future

Reflection:
```

## Model Compatibility

The reflection agent automatically detects and optimizes for different model types:

- **Qwen**: Optimized image processing parameters
- **LLaMA Vision**: Multi-GPU support with specific configurations
- **Phi3V**: Standard VLLM configuration
- **InternVL**: Custom tokenizer support
- **Pixtral**: Standard configuration
- **Default**: Works with any VLLM-compatible model

## Output Format

The agent generates structured reflections that typically include:

- Analysis of error patterns in the negative samples
- Identification of specific reasoning failures
- Clear lessons learned to prevent similar errors
- Actionable insights for improvement

Example output:
```
The negative samples reveal several critical shortcomings in spatial reasoning tasks. First, there is a tendency to focus on only one transformation aspect while ignoring others, as seen in samples that change size but miss color requirements. Second, the samples show confusion about directional concepts, often mixing up left/right or front/behind orientations. Third, there's a pattern of selecting irrelevant transformation functions that don't address the actual changes needed between initial and final states. To avoid these errors, future responses should: carefully analyze all visible changes between images, verify directional transformations against the coordinate system, and ensure all required modifications are included in the final sequence.
```

## Configuration

Key parameters you can adjust:

- `temperature`: Controls randomness of reflection generation (default: 0.7)
- `max_tokens`: Maximum length of reflection (default: 512)
- `max_image_num`: Maximum images per prompt (default: 2)
- `gpu_memory_utilization`: GPU memory usage (default: 0.9)

## Integration

The reflection agent is designed to integrate with the broader Reason-RFT training pipeline:

1. **Training Loop**: Generate reflections during DPO training
2. **Memory Database**: Store reflections for future reference
3. **Evaluation**: Use reflections to understand model weaknesses
4. **Improvement**: Apply lessons learned to enhance training data

## Files

- `reflection_agent.py`: Main implementation
- `test_reflection_agent.py`: Test script and examples
- `README_REFLECTION_AGENT.md`: This documentation

## Dependencies

- `vllm`: For model inference
- `transformers`: For model processing
- `PIL`: For image handling
- `torch`: PyTorch backend

## Notes

- Ensure sufficient GPU memory for the model and image processing
- The agent works best with high-quality negative samples that show clear reasoning errors
- Reflections improve with more diverse negative samples
- The generated reflections can be used to improve future training or prompting strategies
