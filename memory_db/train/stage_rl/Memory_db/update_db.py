"""
Update database with VLM inference results and reward calculations.

This module handles:
1. Multiple VLM inferences for a given VQ (Visual Question) 
2. Reward calculation using multiple reward functions
3. Storing results in the JSON database
"""

import json
import os
import sys
import re
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from database import JSONDatabase
from vlm_agent import create_vlm_agent, get_prompt_template
from on_learning_agent import create_on_learning_agent
from reward import (
    accuracy_reward, 
    math_accuracy_reward, 
    func_accuracy_reward,
    format_reward,
    caption_format_reward,
    reasoning_steps_reward,
    len_reward
)


class VLMInferenceManager:
    """
    Manager class for handling VLM inference and database updates.
    """
    
    def __init__(
        self, 
        model_name_or_path: str,
        db_path: str = "memory_db.json",
        max_image_num: int = 2,
        task_name: str = "trance",
        eval_type: str = "cot-sft",
        enable_on_learning: bool = False
    ):
        """
        Initialize the VLM Inference Manager.
        
        Args:
            model_name_or_path: Path to the VLM model
            db_path: Path to the JSON database file
            max_image_num: Maximum number of images supported
            task_name: Task name for prompt selection
            eval_type: Evaluation type for prompt selection
            enable_on_learning: Whether to enable on-learning agent
        """
        self.db = JSONDatabase(db_path)
        self.vlm_agent = create_vlm_agent(model_name_or_path, max_image_num)
        self.prompt_template = get_prompt_template(task_name, eval_type)
        self.task_name = task_name
        self.eval_type = eval_type
        
        # Initialize on-learning agent if enabled
        self.on_learning_agent = None
        if enable_on_learning:
            self.on_learning_agent = create_on_learning_agent(model_name_or_path, max_image_num)
        
        # Reward function mapping
        self.reward_functions = {
            'accuracy': self._get_accuracy_reward_func(),
            'format': self._get_format_reward_func(),
            'reason': reasoning_steps_reward,
            'length': len_reward
        }
    
    def _get_accuracy_reward_func(self):
        """Get the appropriate accuracy reward function based on task."""
        if self.task_name in ["trance", "trance-left", "trance-right"]:
            return func_accuracy_reward
        elif self.task_name in ["clevr-math", "super-clevr"]:
            return math_accuracy_reward
        else:
            return accuracy_reward
    
    def _get_format_reward_func(self):
        """Get the appropriate format reward function based on eval type."""
        if self.eval_type == "caption-cot":
            return caption_format_reward
        else:
            return format_reward
    
    def multiple_inference(
        self, 
        sample: Dict[str, Any], 
        image_dir: str,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 768
    ) -> List[str]:
        """
        Perform multiple inferences on the same sample.
        
        Args:
            sample: Sample data containing 'image' and 'question'
            image_dir: Directory containing the images
            k: Number of inferences to perform
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of k inference results
        """
        # Update sampling parameters for multiple diverse responses
        self.vlm_agent.sampling_params.temperature = temperature
        self.vlm_agent.sampling_params.max_tokens = max_tokens
        
        responses = []
        for i in range(k):
            try:
                response = self.vlm_agent.generate_single(
                    sample, 
                    image_dir, 
                    self.prompt_template, 
                    self.task_name
                )
                responses.append(response)
                print(f"Inference {i+1}/{k} completed")
            except Exception as e:
                print(f"Error in inference {i+1}: {str(e)}")
                responses.append("")  # Add empty response for failed inference
        
        return responses
    
    def on_learning_inference(
        self,
        sample: Dict[str, Any],
        image_dir: str,
        previous_answer: str = "",
        experience_list: Optional[List[Dict[str, Any]]] = None,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> List[str]:
        """
        Perform multiple inferences using the on-learning agent with experience patterns.
        
        Args:
            sample: Sample data containing 'image' and 'question'
            image_dir: Directory containing the images
            previous_answer: Previous good answer for reference learning
            experience_list: List of experience dictionaries for learning
            k: Number of inferences to perform
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of k inference results from on-learning agent
        """
        if self.on_learning_agent is None:
            raise ValueError("On-learning agent is not enabled. Set enable_on_learning=True in constructor.")
        
        # Update sampling parameters for multiple diverse responses
        self.on_learning_agent.sampling_params.temperature = temperature
        self.on_learning_agent.sampling_params.max_tokens = max_tokens
        
        # Use empty defaults if not provided
        if experience_list is None:
            experience_list = []
        if not previous_answer:
            previous_answer = "No previous answer available."
        
        responses = []
        
        # Prepare batch data for more efficient processing
        batch_data = []
        for i in range(k):
            data = {
                "question": sample.get('question', ''),
                "images": sample.get('image', []),
                "previous_answer": previous_answer,
                "experience_list": experience_list
            }
            batch_data.append(data)
        
        try:
            # Use batch processing for efficiency
            batch_responses = self.on_learning_agent.generate_batch_with_learning(
                batch_data,
                image_dir,
                self.task_name
            )
            responses = batch_responses
            print(f"On-learning batch inference completed: {len(responses)} responses")
            
        except Exception as e:
            print(f"Error in batch on-learning inference: {str(e)}")
            # Fallback to individual inferences
            print("Falling back to individual inferences...")
            
            for i in range(k):
                try:
                    response = self.on_learning_agent.generate_with_learning(
                        question=sample.get('question', ''),
                        images=sample.get('image', []),
                        image_dir=image_dir,
                        previous_answer=previous_answer,
                        experience_list=experience_list,
                        task_name=self.task_name
                    )
                    responses.append(response)
                    print(f"On-learning inference {i+1}/{k} completed")
                    
                except Exception as e:
                    print(f"Error in on-learning inference {i+1}: {str(e)}")
                    responses.append("")  # Add empty response for failed inference
        
        return responses
    
    def calculate_rewards(
        self, 
        responses: List[str], 
        ground_truth: str,
        current_step: int = 0
    ) -> Dict[str, List[float]]:
        """
        Calculate rewards for multiple responses.
        
        Args:
            responses: List of model responses
            ground_truth: Ground truth answer
            current_step: Current training step (for length reward)
            
        Returns:
            Dictionary mapping reward types to lists of scores
        """
        # Format responses for reward functions (they expect specific format)
        formatted_responses = [[[{"content": response}]] for response in responses]
        solutions = [ground_truth] * len(responses)
        
        rewards = {}
        
        # Calculate accuracy rewards
        try:
            accuracy_scores = self.reward_functions['accuracy'](formatted_responses, solutions)
            rewards['accuracy'] = accuracy_scores
        except Exception as e:
            print(f"Error calculating accuracy rewards: {str(e)}")
            rewards['accuracy'] = [0.0] * len(responses)
        
        # Calculate format rewards
        try:
            format_scores = self.reward_functions['format'](formatted_responses)
            rewards['format'] = format_scores
        except Exception as e:
            print(f"Error calculating format rewards: {str(e)}")
            rewards['format'] = [0.0] * len(responses)
        
        # Calculate reasoning rewards
        try:
            reason_scores = self.reward_functions['reason'](formatted_responses)
            rewards['reason'] = reason_scores
        except Exception as e:
            print(f"Error calculating reasoning rewards: {str(e)}")
            rewards['reason'] = [0.0] * len(responses)
        
        # Calculate length rewards
        try:
            length_scores = self.reward_functions['length'](
                formatted_responses, solutions, current_step
            )
            rewards['length'] = length_scores
        except Exception as e:
            print(f"Error calculating length rewards: {str(e)}")
            rewards['length'] = [0.0] * len(responses)
        
        return rewards
    
    def get_relevant_experiences_from_db(
        self,
        current_question: str,
        task_type: Optional[str] = None,
        min_accuracy_score: float = 0.7,
        max_experiences: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant high-quality experiences from the database for on-learning.
        
        Args:
            current_question: The current question to find relevant experiences for
            task_type: Filter by task type (optional)
            min_accuracy_score: Minimum accuracy score to consider as "good" experience
            max_experiences: Maximum number of experiences to return
            
        Returns:
            List of relevant experience dictionaries
        """
        experiences = []
        
        try:
            # Get all training data from database
            all_training_data = self.db.get_all_training_data()
            
            for training_id, training_data in all_training_data.items():
                # Get sorted responses for this training data
                responses = self.db.get_llm_responses_sorted_by_average_score(training_id)
                
                if not responses:
                    continue
                
                # Take the best response (first in sorted list)
                best_response = responses[0]
                accuracy = best_response.get('accuracy', 0)
                
                # Only include responses with high accuracy
                if accuracy >= min_accuracy_score:
                    experience = {
                        "question": training_data.get('problem', ''),
                        "answer": best_response.get('ans', ''),
                        "success": True,
                        "reasoning_pattern": f"High accuracy response (score: {accuracy:.2f})",
                        "scores": {
                            "accuracy": best_response.get('accuracy', 0),
                            "format": best_response.get('format', 0),
                            "reason": best_response.get('reason', 0),
                            "length": best_response.get('length', 0)
                        },
                        "task_type": training_data.get('task_type', task_type)
                    }
                    experiences.append(experience)
            
            # Sort by accuracy score (highest first) and limit to max_experiences
            experiences.sort(key=lambda x: x['scores'].get('accuracy', 0), reverse=True)
            experiences = experiences[:max_experiences]
            
            print(f"Retrieved {len(experiences)} relevant experiences from database")
            
        except Exception as e:
            print(f"Error retrieving experiences from database: {str(e)}")
            experiences = []
        
        return experiences
    
    def on_learning_inference_with_auto_experience(
        self,
        sample: Dict[str, Any],
        image_dir: str,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        min_accuracy_score: float = 0.7,
        max_experiences: int = 3
    ) -> List[str]:
        """
        Perform on-learning inference with automatic experience retrieval from database.
        
        Args:
            sample: Sample data containing 'image' and 'question'
            image_dir: Directory containing the images
            k: Number of inferences to perform
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            min_accuracy_score: Minimum accuracy score for experience selection
            max_experiences: Maximum number of experiences to retrieve
            
        Returns:
            List of k inference results from on-learning agent with auto-retrieved experiences
        """
        if self.on_learning_agent is None:
            raise ValueError("On-learning agent is not enabled. Set enable_on_learning=True in constructor.")
        
        # Automatically retrieve relevant experiences from database
        experience_list = self.get_relevant_experiences_from_db(
            current_question=sample.get('question', ''),
            min_accuracy_score=min_accuracy_score,
            max_experiences=max_experiences
        )
        
        # Get a previous good answer from experiences if available
        previous_answer = ""
        if experience_list:
            # Use the answer from the highest scoring experience
            previous_answer = experience_list[0].get('answer', '')
            print(f"Using previous answer from experience with accuracy: {experience_list[0]['scores']['accuracy']:.2f}")
        else:
            print("No high-quality experiences found in database")
        
        # Perform on-learning inference with retrieved experiences
        return self.on_learning_inference(
            sample=sample,
            image_dir=image_dir,
            previous_answer=previous_answer,
            experience_list=experience_list,
            k=k,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def process_and_store_with_on_learning(
        self,
        training_data: Dict[str, Any],
        image_dir: str,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        current_step: int = 0,
        round: int = 1,
        min_accuracy_score: float = 0.7,
        max_experiences: int = 3
    ) -> bool:
        """
        Process a training sample with on-learning inference and store results.
        
        Args:
            training_data: Training data dict with 'id', 'problem', 'answer', etc.
            image_dir: Directory containing images
            k: Number of inferences to perform
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            current_step: Current training step
            round: Training round number
            min_accuracy_score: Minimum accuracy score for experience selection
            max_experiences: Maximum number of experiences to retrieve
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure round is set in training_data
            if 'round' not in training_data:
                training_data['round'] = round
            
            # First, insert or update the training data in database
            training_id = self.db.insert_training_data(training_data)
            print(f"Training data inserted/updated with ID: {training_id}")
            
            # Prepare sample for on-learning VLM inference
            sample = {
                'image': training_data.get('image', []),
                'question': training_data.get('problem', '')
            }
            
            # Perform on-learning inference with auto experience retrieval
            print(f"Performing {k} on-learning inferences with experience retrieval...")
            responses = self.on_learning_inference_with_auto_experience(
                sample, 
                image_dir, 
                k, 
                temperature, 
                max_tokens,
                min_accuracy_score,
                max_experiences
            )
            
            # Calculate rewards for all responses
            print("Calculating rewards...")
            rewards = self.calculate_rewards(
                responses, 
                training_data.get('answer', ''),
                current_step
            )
            
            # Store each response and its scores in the database
            for i, response in enumerate(responses):
                if response:  # Only store non-empty responses
                    scores = {
                        'accuracy': rewards['accuracy'][i],
                        'format': rewards['format'][i],
                        'reason': rewards['reason'][i],
                        'length': rewards['length'][i]
                    }
                    
                    success = self.db.add_llm_response(
                        training_id,
                        response,
                        scores,
                        reflexion=""  # Empty as requested
                    )
                    
                    if success:
                        print(f"On-learning response {i+1}/{len(responses)} stored successfully")
                        print(f"  Scores: {scores}")
                    else:
                        print(f"Failed to store on-learning response {i+1}")
            
            return True
            
        except Exception as e:
            print(f"Error in process_and_store_with_on_learning: {str(e)}")
            return False
    
    def process_and_store(
        self,
        training_data: Dict[str, Any],
        image_dir: str,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 768,
        current_step: int = 0,
        round: int = 1
    ) -> bool:
        """
        Process a training sample with multiple inferences and store results.
        
        Args:
            training_data: Training data dict with 'id', 'problem', 'answer', etc.
            image_dir: Directory containing images
            k: Number of inferences to perform
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            current_step: Current training step
            round: Training round number
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure round is set in training_data
            if 'round' not in training_data:
                training_data['round'] = round
            
            # First, insert or update the training data in database
            training_id = self.db.insert_training_data(training_data)
            print(f"Training data inserted/updated with ID: {training_id}")
            
            # Prepare sample for VLM inference
            sample = {
                'image': training_data.get('image', []),
                'question': training_data.get('problem', '')
            }
            
            # Perform multiple inferences
            print(f"Performing {k} inferences...")
            responses = self.multiple_inference(
                sample, image_dir, k, temperature, max_tokens
            )
            
            # Calculate rewards for all responses
            print("Calculating rewards...")
            rewards = self.calculate_rewards(
                responses, 
                training_data.get('answer', ''),
                current_step
            )
            
            # Store each response and its scores in the database
            for i, response in enumerate(responses):
                if response:  # Only store non-empty responses
                    scores = {
                        'accuracy': rewards['accuracy'][i],
                        'format': rewards['format'][i],
                        'reason': rewards['reason'][i],
                        'length': rewards['length'][i]
                    }
                    
                    success = self.db.add_llm_response(
                        training_id,
                        response,
                        scores,
                        reflexion=""  # Empty as requested
                    )
                    
                    if success:
                        print(f"Response {i+1}/{len(responses)} stored successfully")
                        print(f"  Scores: {scores}")
                    else:
                        print(f"Failed to store response {i+1}")
            
            return True
            
        except Exception as e:
            print(f"Error in process_and_store: {str(e)}")
            return False
    
    def process_batch(
        self,
        training_data_list: List[Dict[str, Any]],
        image_dir: str,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 768,
        current_step: int = 0,
        round: int = 1
    ) -> Dict[str, bool]:
        """
        Process a batch of training samples.
        
        Args:
            training_data_list: List of training data dictionaries
            image_dir: Directory containing images
            k: Number of inferences per sample
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            current_step: Current training step
            round: Training round number
            
        Returns:
            Dictionary mapping training IDs to success status
        """
        results = {}
        
        for i, training_data in enumerate(training_data_list):
            print(f"\nProcessing sample {i+1}/{len(training_data_list)}")
            print(f"ID: {training_data.get('id', 'unknown')}")
            
            success = self.process_and_store(
                training_data,
                image_dir,
                k,
                temperature,
                max_tokens,
                current_step,
                round
            )
            
            results[training_data.get('id', f'sample_{i}')] = success
        
        return results
    
    def process_batch_with_on_learning(
        self,
        training_data_list: List[Dict[str, Any]],
        image_dir: str,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        current_step: int = 0,
        round: int = 1,
        min_accuracy_score: float = 0.7,
        max_experiences: int = 3
    ) -> Dict[str, bool]:
        """
        Process a batch of training samples using on-learning inference.
        
        Args:
            training_data_list: List of training data dictionaries
            image_dir: Directory containing images
            k: Number of inferences per sample
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            current_step: Current training step
            round: Training round number
            min_accuracy_score: Minimum accuracy score for experience selection
            max_experiences: Maximum number of experiences to retrieve
            
        Returns:
            Dictionary mapping training IDs to success status
        """
        results = {}
        
        for i, training_data in enumerate(training_data_list):
            print(f"\nProcessing sample {i+1}/{len(training_data_list)} with on-learning")
            print(f"ID: {training_data.get('id', 'unknown')}")
            
            success = self.process_and_store_with_on_learning(
                training_data,
                image_dir,
                k,
                temperature,
                max_tokens,
                current_step,
                round,
                min_accuracy_score,
                max_experiences
            )
            
            results[training_data.get('id', f'sample_{i}')] = success
        
        return results


def load_training_data_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load training data from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing training data
        
    Returns:
        List of training data dictionaries
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if 'training_data' in data:
            return list(data['training_data'].values())
        elif 'data' in data:
            return data['data']
        else:
            return [data]  # Single sample
    else:
        raise ValueError("Unsupported JSON format")


def create_sample_vq_data() -> Dict[str, Any]:
    """
    Create a sample VQ data for testing.
    
    Returns:
        Sample training data dictionary
    """
    return {
        'id': 'test-vq-001',
        'problem': 'What is the spatial transformation applied to the object in these images? The initial objects are: [("obj1", "cube", "medium", "red", "metal")]',
        'answer': 'change_color(obj1, blue)',
        'round': 1,
        'image': [
            {
                'path': 'initial_state.png',
                'type': 'Spatial-Transformation',
                'transformation': 'color_change',
                'description': 'Initial state image'
            },
            {
                'path': 'final_state.png', 
                'type': 'Spatial-Transformation',
                'transformation': 'color_change',
                'description': 'Final state image'
            }
        ]
    }


# Example usage and main function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM Inference and Database Update")
    parser.add_argument('--model_path', type=str, required=True, help="Path to VLM model")
    parser.add_argument('--db_path', type=str, default="memory_db.json", help="Database file path")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing images")
    parser.add_argument('--data_json', type=str, help="JSON file with training data")
    parser.add_argument('--k', type=int, default=5, help="Number of inferences per sample")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature")
    parser.add_argument('--max_tokens', type=int, default=768, help="Maximum tokens")
    parser.add_argument('--task_name', type=str, default="trance", help="Task name")
    parser.add_argument('--eval_type', type=str, default="cot-sft", help="Evaluation type")
    parser.add_argument('--current_step', type=int, default=0, help="Current training step")
    parser.add_argument('--round', type=int, default=1, help="Training round number")
    parser.add_argument('--test_sample', action='store_true', help="Use test sample data")
    parser.add_argument('--enable_on_learning', action='store_true', help="Enable on-learning agent")
    parser.add_argument('--min_accuracy_score', type=float, default=0.7, help="Minimum accuracy score for experience selection")
    parser.add_argument('--max_experiences', type=int, default=3, help="Maximum number of experiences to retrieve")
    
    args = parser.parse_args()
    
    # Initialize the manager
    manager = VLMInferenceManager(
        model_name_or_path=args.model_path,
        db_path=args.db_path,
        max_image_num=2,
        task_name=args.task_name,
        eval_type=args.eval_type,
        enable_on_learning=args.enable_on_learning
    )
    
    if args.test_sample:
        # Use test sample
        print("Using test sample data...")
        sample_data = create_sample_vq_data()
        
        if args.enable_on_learning:
            print("Using on-learning agent for test sample...")
            success = manager.process_and_store_with_on_learning(
                sample_data,
                args.image_dir,
                args.k,
                args.temperature,
                args.max_tokens,
                args.current_step,
                args.round,
                args.min_accuracy_score,
                args.max_experiences
            )
        else:
            success = manager.process_and_store(
                sample_data,
                args.image_dir,
                args.k,
                args.temperature,
                args.max_tokens,
                args.current_step,
                args.round
            )
        print(f"Test sample processing {'succeeded' if success else 'failed'}")
        
    elif args.data_json:
        # Load and process data from JSON file
        print(f"Loading training data from {args.data_json}...")
        training_data_list = load_training_data_from_json(args.data_json)
        print(f"Loaded {len(training_data_list)} training samples")
        
        if args.enable_on_learning:
            print("Using on-learning agent for batch processing...")
            results = manager.process_batch_with_on_learning(
                training_data_list,
                args.image_dir,
                args.k,
                args.temperature,
                args.max_tokens,
                args.current_step,
                args.round,
                args.min_accuracy_score,
                args.max_experiences
            )
        else:
            results = manager.process_batch(
                training_data_list,
                args.image_dir,
                args.k,
                args.temperature,
                args.max_tokens,
                args.current_step,
                args.round
            )
        
        # Print summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        inference_type = "on-learning" if args.enable_on_learning else "standard"
        print(f"\n{inference_type.capitalize()} processing complete: {successful}/{total} samples succeeded")
        
        # Show failed samples
        failed = [sample_id for sample_id, success in results.items() if not success]
        if failed:
            print(f"Failed samples: {failed}")
            
    else:
        print("Please provide either --data_json or --test_sample")
        print("Example usage:")
        print("  Standard inference:")
        print("    python update_db.py --model_path /path/to/qwen_model --image_dir /path/to/images --data_json training_data.json --round 2")
        print("    python update_db.py --model_path /path/to/qwen_model --image_dir /path/to/images --test_sample --round 1")
        print("  On-learning inference:")
        print("    python update_db.py --model_path /path/to/qwen_model --image_dir /path/to/images --data_json training_data.json --enable_on_learning --round 2")
        print("    python update_db.py --model_path /path/to/qwen_model --image_dir /path/to/images --test_sample --enable_on_learning --max_experiences 5")
