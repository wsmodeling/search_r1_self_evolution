import json
import os
from typing import List, Dict, Any, Union, Optional
from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
import argparse

# ====================================
#  ON-LEARNING AGENT PROMPTS
# ====================================

ON_LEARNING_TRANCE_PROMPT = '''You are an intelligent spatial visual reasoning agent that learns from previous experiences and good examples.

Your task is to complete the spatial visual reasoning task according to the following rules:

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.

The **attributes of the initial objects** are provided as a list of tuples in the following format:
**('object_id', 'shape', 'size', 'color', 'material')**
Each tuple represents an object and its properties in the initial state.

The transformation should be a sequence of functions with a length ranging from 1 to 4, where each function is represented as **'func(object_id, value)'**.

### Available functions and values:

1. **'change_size(object_id, value)'** - Changes the object to a new size relative to its initial size.
   - Possible values: `['small', 'medium', 'large']`

2. **'change_color(object_id, value)'** - Changes the object to a new color relative to its initial color.
   - Possible values: `['yellow', 'gray', 'cyan', 'blue', 'brown', 'green', 'red', 'purple']`

3. **'change_material(object_id, value)'** - Changes the object to a new material relative to its initial material.
   - Possible values: `['glass', 'metal', 'rubber']`

4. **'change_shape(object_id, value)'** - Changes the object to a new shape relative to its initial shape.
   - Possible values: `['cube', 'sphere', 'cylinder']`

5. **'change_position(object_id, value)'** - Moves the object to a new position relative to its initial location.
   - Possible values: `['front', 'behind', 'left', 'right', 'front_left', 'front_right', 'behind_left', 'behind_right']`
   - 'front' means moving forward along the object's initial direction.
   - 'behind' means moving backward along the object's initial direction.
   - 'left' means moving to the left of the object's initial orientation.
   - 'right' means moving to the right of the object's initial orientation.
   - 'front_left' means moving diagonally toward the front and left of the initial location.
   - 'front_right' means moving diagonally toward the front and right of the initial location.
   - 'behind_left' means moving diagonally toward the behind and left of the initial location.
   - 'behind_right' means moving diagonally toward the behind and right of the initial location.

### Learning from Experience:

You have access to previous good examples and experience patterns:

**Previous Good Answer:**
{previous_answer}

**Experience Patterns:**
{experience_list}

Use these examples to understand successful reasoning patterns and apply similar logic to the current problem.

### Output Format:

You should provide a comprehensive analysis and then give your answer:

- **Experience Analysis**: Analyze how the previous good answer and experience patterns relate to the current problem, enclosed within `<experience>...</experience>`
- **Reasoning Process**: Provide step-by-step logical reasoning, learning from the patterns you observed, enclosed within `<think>...</think>`
- **Final Answer**: The sequence of functions only, enclosed within `<answer>...</answer>`

Now, solve this problem:

{question}

Please analyze the experience, think through the problem, and provide your final answer in the specified format.
'''

ON_LEARNING_CLEVR_MATH_PROMPT = '''You are an intelligent mathematical reasoning agent that learns from previous experiences and good examples.

**Previous Good Answer:**
{previous_answer}

**Experience Patterns:**
{experience_list}

Use these examples to understand successful reasoning patterns and mathematical problem-solving approaches.

### Output Format:

- **Experience Analysis**: Analyze how the previous good answer and experience patterns relate to the current problem, enclosed within `<experience>...</experience>`
- **Reasoning Process**: Provide step-by-step mathematical reasoning, learning from the patterns you observed, enclosed within `<think>...</think>`
- **Final Answer**: The numerical answer only, enclosed within `<answer>...</answer>`

{question}

Please analyze the experience, think through the problem, and provide your final answer in the specified format.
'''

ON_LEARNING_GEOMETRY_PROMPT = '''You are an intelligent geometry reasoning agent that learns from previous experiences and good examples.

**Previous Good Answer:**
{previous_answer}

**Experience Patterns:**
{experience_list}

Use these examples to understand successful reasoning patterns and geometric problem-solving approaches.

### Output Format:

- **Experience Analysis**: Analyze how the previous good answer and experience patterns relate to the current problem, enclosed within `<experience>...</experience>`
- **Reasoning Process**: Provide step-by-step geometric reasoning, learning from the patterns you observed, enclosed within `<think>...</think>`
- **Final Answer**: The answer (number or choice), enclosed within `<answer>...</answer>`

{question}

Please analyze the experience, think through the problem, and provide your final answer in the specified format.
'''


class OnLearningAgent:
    """
    An agent that learns from previous good answers and experience patterns
    to improve its performance on visual reasoning tasks.
    """
    
    def __init__(self, model_name_or_path: str, max_image_num: int = 2):
        """
        Initialize the on-learning agent.
        
        Args:
            model_name_or_path: Path to the model checkpoint
            max_image_num: Maximum number of images to process
        """
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=1024,  # Increased for detailed reasoning
        )
        
        self.model_name_or_path = model_name_or_path
        self.experience_memory = []
    
    def format_experience_list(self, experience_list: List[Dict[str, Any]]) -> str:
        """
        Format the experience list into a readable string for the prompt.
        
        Args:
            experience_list: List of experience dictionaries
            
        Returns:
            Formatted experience string
        """
        if not experience_list:
            return "No previous experience available."
        
        formatted_experiences = []
        for i, exp in enumerate(experience_list, 1):
            exp_str = f"Experience {i}:\n"
            exp_str += f"  Question: {exp.get('question', 'N/A')}\n"
            exp_str += f"  Answer: {exp.get('answer', 'N/A')}\n"
            exp_str += f"  Success: {exp.get('success', 'Unknown')}\n"
            exp_str += f"  Reasoning Pattern: {exp.get('reasoning_pattern', 'N/A')}\n"
            formatted_experiences.append(exp_str)
        
        return "\n".join(formatted_experiences)
    
    def get_prompt_template(self, task_name: str) -> str:
        """
        Get the appropriate prompt template based on task name.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Prompt template string
        """
        if task_name in ["trance", "trance-left", "trance-right"]:
            return ON_LEARNING_TRANCE_PROMPT
        elif task_name in ["clevr-math", "super-clevr"]:
            return ON_LEARNING_CLEVR_MATH_PROMPT
        elif task_name in ["geomath", "geometry3k"]:
            return ON_LEARNING_GEOMETRY_PROMPT
        else:
            # Default to trance prompt
            return ON_LEARNING_TRANCE_PROMPT
    
    def generate_with_learning(
        self,
        question: str,
        images: Union[List[str], str],
        image_dir: str,
        previous_answer: str,
        experience_list: List[Dict[str, Any]],
        task_name: str = "trance"
    ) -> str:
        """
        Generate an answer using previous good answer and experience patterns.
        
        Args:
            question: The question to answer
            images: Image file paths (list or single string)
            image_dir: Directory containing images
            previous_answer: Previous good answer for reference
            experience_list: List of experience dictionaries
            task_name: Task name for prompt selection
            
        Returns:
            Generated response text
        """
        # Process images
        image_objects = []
        if isinstance(images, list):
            for img_path in images:
                image_full_path = os.path.join(image_dir, img_path)
                if os.path.exists(image_full_path):
                    image_objects.append(Image.open(image_full_path))
                else:
                    print(f"Warning: Image not found: {image_full_path}")
        else:
            image_full_path = os.path.join(image_dir, images)
            if os.path.exists(image_full_path):
                image_objects.append(Image.open(image_full_path))
            else:
                print(f"Warning: Image not found: {image_full_path}")
        
        # Get prompt template and format it
        prompt_template = self.get_prompt_template(task_name)
        formatted_experience = self.format_experience_list(experience_list)
        
        formatted_prompt = prompt_template.format(
            question=question,
            previous_answer=previous_answer,
            experience_list=formatted_experience
        )
        
        # Create conversation messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_prompt}
                ] + [{"type": "image", "image": img} for img in image_objects]
            }
        ]
        
        # Apply chat template
        vllm_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Generate response
        prompt_with_images = {
            "prompt": vllm_prompt,
            "multi_modal_data": {"image": image_objects}
        }
        
        outputs = self.model.generate([prompt_with_images], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return generated_text
    
    def generate_batch_with_learning(
        self,
        batch_data: List[Dict[str, Any]],
        image_dir: str,
        task_name: str = "trance"
    ) -> List[str]:
        """
        Generate responses for a batch of samples with learning.
        
        Args:
            batch_data: List of dictionaries containing question, images, previous_answer, experience_list
            image_dir: Directory containing images
            task_name: Task name for prompt selection
            
        Returns:
            List of generated responses
        """
        prompts_text_and_vision = []
        
        for data in batch_data:
            question = data["question"]
            images = data["images"]
            previous_answer = data.get("previous_answer", "No previous answer available.")
            experience_list = data.get("experience_list", [])
            
            # Process images
            image_objects = []
            if isinstance(images, list):
                for img_path in images:
                    image_full_path = os.path.join(image_dir, img_path)
                    if os.path.exists(image_full_path):
                        image_objects.append(Image.open(image_full_path))
            else:
                image_full_path = os.path.join(image_dir, images)
                if os.path.exists(image_full_path):
                    image_objects.append(Image.open(image_full_path))
            
            # Get prompt template and format it
            prompt_template = self.get_prompt_template(task_name)
            formatted_experience = self.format_experience_list(experience_list)
            
            formatted_prompt = prompt_template.format(
                question=question,
                previous_answer=previous_answer,
                experience_list=formatted_experience
            )
            
            # Create conversation messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": formatted_prompt}
                    ] + [{"type": "image", "image": img} for img in image_objects]
                }
            ]
            
            # Apply chat template
            vllm_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Add to batch
            prompts_text_and_vision.append(
                {
                    "prompt": vllm_prompt,
                    "multi_modal_data": {"image": image_objects}
                }
            )
        
        # Generate responses for the batch
        outputs = self.model.generate(prompts_text_and_vision, sampling_params=self.sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        
        return results
    
    def add_experience(
        self,
        question: str,
        answer: str,
        success: bool,
        reasoning_pattern: Optional[str] = None
    ):
        """
        Add a new experience to the agent's memory.
        
        Args:
            question: The question that was asked
            answer: The answer that was given
            success: Whether the answer was successful
            reasoning_pattern: Pattern or strategy used in reasoning
        """
        experience = {
            "question": question,
            "answer": answer,
            "success": success,
            "reasoning_pattern": reasoning_pattern or "Standard reasoning"
        }
        self.experience_memory.append(experience)
    
    def get_relevant_experiences(
        self,
        current_question: str,
        max_experiences: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get relevant experiences based on the current question.
        This is a simple implementation - could be enhanced with semantic similarity.
        
        Args:
            current_question: The current question to find relevant experiences for
            max_experiences: Maximum number of experiences to return
            
        Returns:
            List of relevant experience dictionaries
        """
        # Simple heuristic: return successful experiences first
        successful_experiences = [exp for exp in self.experience_memory if exp.get("success", False)]
        return successful_experiences[-max_experiences:]  # Return most recent successful experiences
    
    def save_experience_memory(self, filepath: str):
        """Save experience memory to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.experience_memory, f, indent=2)
    
    def load_experience_memory(self, filepath: str):
        """Load experience memory from a JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.experience_memory = json.load(f)


class QwenOnLearningAgent(OnLearningAgent):
    """Qwen-specific implementation of OnLearningAgent"""
    
    def __init__(self, model_name_or_path: str, max_image_num: int = 2, min_pixels: int = 3136, max_pixels: int = 480000):
        super().__init__(model_name_or_path, max_image_num)
        try:
            self.processor.image_processor.min_pixels = min_pixels
            self.processor.image_processor.max_pixels = max_pixels
        except:
            pass


class MllamaOnLearningAgent(OnLearningAgent):
    """Mllama-specific implementation of OnLearningAgent"""
    
    def __init__(self, model_name_or_path: str, max_image_num: int = 2):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=2,
            max_model_len=4096,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
            max_num_seqs=16,
            enforce_eager=True,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=1024,
        )
        
        self.model_name_or_path = model_name_or_path
        self.experience_memory = []


def create_on_learning_agent(model_name_or_path: str, max_image_num: int = 2) -> OnLearningAgent:
    """
    Factory function to create the appropriate OnLearningAgent based on model name.
    
    Args:
        model_name_or_path: Path to the model checkpoint
        max_image_num: Maximum number of images to process
        
    Returns:
        OnLearningAgent instance
    """
    print(f"Loading On-Learning Agent from {model_name_or_path} ...")
    
    if 'qwen' in model_name_or_path.lower():
        print("======== Using QwenOnLearningAgent ==========")
        return QwenOnLearningAgent(model_name_or_path, max_image_num)
    elif 'llama' in model_name_or_path.lower() and 'vision' in model_name_or_path.lower():
        print("======== Using MllamaOnLearningAgent ==========")
        return MllamaOnLearningAgent(model_name_or_path, max_image_num)
    else:
        print("======== Using Default OnLearningAgent ==========")
        return OnLearningAgent(model_name_or_path, max_image_num)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="On-Learning Agent for visual question answering with experience learning.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--task_name', type=str, default="trance", help="Task name (default: trance)")
    parser.add_argument('--max_image_num', type=int, default=2, help="Maximum number of images (default: 2)")
    parser.add_argument('--experience_file', type=str, help="Path to experience memory file")
    
    args = parser.parse_args()
    
    # Create on-learning agent
    agent = create_on_learning_agent(args.model_name_or_path, args.max_image_num)
    
    # Load experience memory if provided
    if args.experience_file and os.path.exists(args.experience_file):
        agent.load_experience_memory(args.experience_file)
        print(f"Loaded {len(agent.experience_memory)} experiences from {args.experience_file}")
    
    print(f"On-Learning Agent created successfully!")
    print(f"Model: {args.model_name_or_path}")
    print(f"Task: {args.task_name}")
    
    # Example of how to use the agent
    # question = "Your question here"
    # images = ["image1.jpg", "image2.jpg"]  # or single image "image.jpg"
    # image_dir = "/path/to/images"
    # previous_answer = "Previous good answer example"
    # experience_list = [
    #     {
    #         "question": "Similar question",
    #         "answer": "Good answer",
    #         "success": True,
    #         "reasoning_pattern": "Step-by-step spatial analysis"
    #     }
    # ]
    # 
    # response = agent.generate_with_learning(
    #     question=question,
    #     images=images,
    #     image_dir=image_dir,
    #     previous_answer=previous_answer,
    #     experience_list=experience_list,
    #     task_name=args.task_name
    # )
    # print(f"Response: {response}")
    # 
    # # Add this experience to memory
    # agent.add_experience(
    #     question=question,
    #     answer=response,
    #     success=True,  # You would evaluate this
    #     reasoning_pattern="Learned from previous examples"
    # )