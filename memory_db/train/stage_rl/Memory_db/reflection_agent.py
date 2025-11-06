import json
import os
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


class ReflectionAgent:
    """
    Reflection Agent that analyzes poor LLM outputs and generates reflections
    to help improve future performance by summarizing shortcomings and lessons learned.
    """
    
    def __init__(self, model_name_or_path: str, max_image_num: int = 2):
        """
        Initialize the Reflection Agent
        
        Args:
            model_name_or_path: Path to the language model
            max_image_num: Maximum number of images to process
        """
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        
        # Handle different model types (similar to vlm_agent.py)
        if 'qwen' in model_name_or_path.lower():
            try:
                self.processor.image_processor.min_pixels = 3136
                self.processor.image_processor.max_pixels = 480000
            except:
                pass
        
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": max_image_num},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,  # Slightly higher temperature for more diverse reflections
            top_p=0.9,
            top_k=50,
            max_tokens=512,  # Enough for a comprehensive reflection paragraph
        )
        
        self.model_name_or_path = model_name_or_path
    
    def generate_reflection(
        self, 
        image_path: str, 
        correct_answer: str, 
        negative_samples: List[str],
        task_context: Optional[str] = None
    ) -> str:
        """
        Generate a reflection based on an image, correct answer, and negative samples
        
        Args:
            image_path: Path to the input image
            correct_answer: The correct answer for the task
            negative_samples: List of poor/incorrect LLM outputs
            task_context: Optional context about the task (e.g., "spatial transformation", "math reasoning")
            
        Returns:
            Generated reflection as a string
        """
        # Load and process the image
        image = None
        if os.path.exists(image_path):
            image = Image.open(image_path)
        
        # Format negative samples for the prompt
        negative_samples_text = "\n".join([f"Sample {i+1}: {sample}" for i, sample in enumerate(negative_samples)])
        
        # Create the reflection prompt
        task_desc = f" for {task_context}" if task_context else ""
        prompt = f"""Given the negative samples{task_desc} below, summarize the shortcomings of these negative samples and the lessons learned on how to avoid generating similar ones into a single paragraph. Translate this entire passage into English.

Correct Answer: {correct_answer}

Negative Samples:
{negative_samples_text}

Please analyze these negative samples and provide a comprehensive reflection that identifies:
1. Common patterns of errors
2. Specific shortcomings in reasoning or approach
3. Key lessons to avoid similar mistakes in the future

Reflection:"""

        # Create conversation messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + ([{"type": "image", "image": image}] if image else [])
            }
        ]

        # Apply chat template
        vllm_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate response
        prompt_with_images = {
            "prompt": vllm_prompt,
            "multi_modal_data": {"image": [image] if image else []}
        }
        
        outputs = self.model.generate([prompt_with_images], sampling_params=self.sampling_params, use_tqdm=False)
        reflection = outputs[0].outputs[0].text.strip()
        
        return reflection
    
    def generate_batch_reflections(
        self, 
        reflection_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate reflections for a batch of samples
        
        Args:
            reflection_data: List of dictionaries containing:
                - image_path: path to image
                - correct_answer: correct answer
                - negative_samples: list of poor outputs
                - task_context: optional task context
                
        Returns:
            List of generated reflections
        """
        prompts_with_images = []
        
        for data in reflection_data:
            image_path = data["image_path"]
            correct_answer = data["correct_answer"]
            negative_samples = data["negative_samples"]
            task_context = data.get("task_context", None)
            
            # Load and process the image
            image = None
            if os.path.exists(image_path):
                image = Image.open(image_path)
            
            # Format negative samples for the prompt
            negative_samples_text = "\n".join([f"Sample {i+1}: {sample}" for i, sample in enumerate(negative_samples)])
            
            # Create the reflection prompt
            task_desc = f" for {task_context}" if task_context else ""
            prompt = f"""Given the negative samples{task_desc} below, summarize the shortcomings of these negative samples and the lessons learned on how to avoid generating similar ones into a single paragraph. Translate this entire passage into English.

Correct Answer: {correct_answer}

Negative Samples:
{negative_samples_text}

Please analyze these negative samples and provide a comprehensive reflection that identifies:
1. Common patterns of errors
2. Specific shortcomings in reasoning or approach
3. Key lessons to avoid similar mistakes in the future

Reflection:"""

            # Create conversation messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + ([{"type": "image", "image": image}] if image else [])
                }
            ]

            # Apply chat template
            vllm_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Add to batch
            prompts_with_images.append({
                "prompt": vllm_prompt,
                "multi_modal_data": {"image": [image] if image else []}
            })

        # Generate responses for the batch
        outputs = self.model.generate(prompts_with_images, sampling_params=self.sampling_params, use_tqdm=False)

        # Extract reflections
        reflections = []
        for output in outputs:
            reflection = output.outputs[0].text.strip()
            reflections.append(reflection)

        return reflections
    
    def save_reflection_results(
        self, 
        results: List[Dict[str, Any]], 
        output_file: str
    ) -> None:
        """
        Save reflection results to a JSON file
        
        Args:
            results: List of dictionaries containing reflection data and results
            output_file: Path to output JSON file
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Reflection results saved to: {output_file}")


def create_reflection_agent(model_name_or_path: str, max_image_num: int = 2) -> ReflectionAgent:
    """
    Factory function to create a reflection agent
    
    Args:
        model_name_or_path: Path to the language model
        max_image_num: Maximum number of images to process
        
    Returns:
        ReflectionAgent instance
    """
    print(f"Creating Reflection Agent with model: {model_name_or_path}")
    return ReflectionAgent(model_name_or_path, max_image_num)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reflection Agent for analyzing poor LLM outputs")
    parser.add_argument('--model_name_or_path', type=str, required=True, 
                       help="Path to the model checkpoint.")
    parser.add_argument('--image_path', type=str, required=True,
                       help="Path to the input image")
    parser.add_argument('--correct_answer', type=str, required=True,
                       help="The correct answer")
    parser.add_argument('--negative_samples', type=str, nargs='+', required=True,
                       help="List of negative/poor samples")
    parser.add_argument('--task_context', type=str, default=None,
                       help="Optional task context")
    parser.add_argument('--output_file', type=str, default="reflection_output.json",
                       help="Output file for saving results")
    
    args = parser.parse_args()
    
    # Create reflection agent
    reflection_agent = create_reflection_agent(args.model_name_or_path)
    
    # Generate reflection
    reflection = reflection_agent.generate_reflection(
        image_path=args.image_path,
        correct_answer=args.correct_answer,
        negative_samples=args.negative_samples,
        task_context=args.task_context
    )
    
    print("Generated Reflection:")
    print("-" * 50)
    print(reflection)
    print("-" * 50)
    
    # Save results
    results = [{
        "image_path": args.image_path,
        "correct_answer": args.correct_answer,
        "negative_samples": args.negative_samples,
        "task_context": args.task_context,
        "reflection": reflection
    }]
    
    reflection_agent.save_reflection_results(results, args.output_file)
    
    print(f"\nReflection completed and saved to {args.output_file}")