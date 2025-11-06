# On-Learning Agent

The On-Learning Agent is an intelligent visual reasoning agent that learns from previous experiences and good examples to improve its performance over time.

## Features

- **Experience-based Learning**: Uses previous good answers and experience patterns to inform new responses
- **Multi-modal Support**: Handles both text and visual inputs (images)
- **Memory Management**: Maintains and uses experience memory across sessions
- **Multiple Task Support**: Supports spatial transformation, mathematical reasoning, and geometry tasks
- **Batch Processing**: Can process multiple samples efficiently

## Architecture

The agent consists of several key components:

1. **OnLearningAgent**: Base class with core functionality
2. **QwenOnLearningAgent**: Qwen-specific implementation
3. **MllamaOnLearningAgent**: Mllama-specific implementation
4. **Experience Memory**: Stores and retrieves past experiences
5. **Prompt Templates**: Task-specific prompts that incorporate learning

## Usage

### Basic Usage

```python
from on_learning_agent import create_on_learning_agent

# Create agent
agent = create_on_learning_agent("path/to/model", max_image_num=2)

# Generate response with learning
response = agent.generate_with_learning(
    question="Your question here",
    images=["image1.jpg", "image2.jpg"],
    image_dir="/path/to/images",
    previous_answer="Previous good answer example",
    experience_list=[
        {
            "question": "Similar question",
            "answer": "Good answer",
            "success": True,
            "reasoning_pattern": "Step-by-step spatial analysis"
        }
    ],
    task_name="trance"
)
```

### Command Line Usage

```bash
python on_learning_agent.py \
    --model_name_or_path "/path/to/model" \
    --task_name "trance" \
    --max_image_num 2 \
    --experience_file "experience_memory.json"
```

### Experience Management

```python
# Add new experience
agent.add_experience(
    question="What transformation occurred?",
    answer="change_color(obj1, red)",
    success=True,
    reasoning_pattern="Color change detection"
)

# Save experience memory
agent.save_experience_memory("experience_memory.json")

# Load experience memory
agent.load_experience_memory("experience_memory.json")

# Get relevant experiences
relevant_exp = agent.get_relevant_experiences("current question", max_experiences=5)
```

### Batch Processing

```python
batch_data = [
    {
        "question": "Question 1",
        "images": ["img1.jpg", "img2.jpg"],
        "previous_answer": "Previous answer 1",
        "experience_list": [...]
    },
    {
        "question": "Question 2", 
        "images": ["img3.jpg"],
        "previous_answer": "Previous answer 2",
        "experience_list": [...]
    }
]

responses = agent.generate_batch_with_learning(
    batch_data=batch_data,
    image_dir="/path/to/images",
    task_name="trance"
)
```

## Supported Tasks

### 1. Spatial Transformation (`trance`, `trance-left`, `trance-right`)

Analyzes spatial transformations between images using functions like:
- `change_size(object_id, value)`
- `change_color(object_id, value)`
- `change_material(object_id, value)`
- `change_shape(object_id, value)`
- `change_position(object_id, value)`

### 2. Mathematical Reasoning (`clevr-math`, `super-clevr`)

Solves mathematical problems based on visual scenes.

### 3. Geometry (`geomath`, `geometry3k`)

Handles geometric reasoning and problem-solving.

## Input Format

### Question
Standard text question string.

### Images
- Single image: `"image.jpg"`
- Multiple images: `["image1.jpg", "image2.jpg"]`

### Previous Answer
A string containing a previous good answer that serves as a learning example.

### Experience List
List of dictionaries with the following structure:
```python
{
    "question": "Previous question text",
    "answer": "Previous answer",
    "success": True/False,
    "reasoning_pattern": "Description of reasoning approach"
}
```

## Output Format

The agent provides structured output with three main sections:

1. **Experience Analysis** (`<experience>...</experience>`): How previous examples relate to current problem
2. **Reasoning Process** (`<think>...</think>`): Step-by-step logical reasoning
3. **Final Answer** (`<answer>...</answer>`): The actual answer

## Model Support

The agent automatically detects and configures for different model types:

- **Qwen Models**: Uses `QwenOnLearningAgent` with pixel configuration
- **Mllama Models**: Uses `MllamaOnLearningAgent` with tensor parallelism
- **Other Models**: Uses base `OnLearningAgent`

## Configuration

### Sampling Parameters
- Temperature: 0.1 (low for consistency)
- Top-p: 0.9
- Top-k: 50
- Max tokens: 1024 (for detailed reasoning)

### Model Parameters
- GPU memory utilization: 0.9 (0.8 for Mllama)
- Enable prefix caching: True
- Trust remote code: True

## Example Experience Entry

```python
experience = {
    "question": "Given two images showing a cube changing from red to blue, what transformation occurred?",
    "answer": "change_color(cube1, blue)",
    "success": True,
    "reasoning_pattern": "Visual comparison of object properties between initial and final states"
}
```

## Best Practices

1. **Quality Experience Data**: Ensure experience entries contain successful examples with clear reasoning patterns
2. **Relevant Context**: Provide previous answers that are similar in structure or domain to the current question
3. **Memory Management**: Regularly save experience memory to preserve learning across sessions
4. **Task-specific Usage**: Use appropriate task names to get optimized prompts
5. **Batch Processing**: Use batch processing for efficiency when handling multiple samples

## Error Handling

The agent includes robust error handling for:
- Missing image files (warnings logged)
- Empty experience lists (fallback messages)
- Model loading issues (automatic fallback to base agent)

## Future Enhancements

Potential improvements include:
- Semantic similarity matching for experience retrieval
- Confidence scoring for generated answers
- Dynamic prompt adaptation based on success rates
- Multi-modal experience embedding
