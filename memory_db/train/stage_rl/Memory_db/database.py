"""
JSON-based database for Reason-RFT training data with LLM answers and scoring.

This module provides a simple JSON file-based database for storing training data,
including spatial transformation information, LLM responses, and evaluation scores.
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
from copy import deepcopy



class JSONDatabase:
    """
    JSON file-based database for storing training data with LLM responses and scores.
    
    Data structure:
    {
        "training_data": {
            "trance-001": {
                "id": "trance-001",
                "problem": "What is the spatial transformation?",
                "answer": "rotation by 90 degrees",
                "round": 1,
                "image": [...],  # List of image info
                "experience": [],  # List of experience strings
                "llm_answers_and_score": [
                    {
                        "ans": "response text",
                        "reflexion": "reflection on the reasoning process",
                        "accuracy": 0.85,
                        "format": 0.90,
                        "reason": 0.75,
                        "length": 0.80
                    }
                ]
            }
        }
    }
    """
    
    def __init__(self, db_path: str = "memory_db.json"):
        """
        Initialize the JSON database.
        
        Args:
            db_path: Path to the JSON database file
        """
        self.db_path = Path(db_path)
        self.lock = threading.Lock()  # For thread safety
        self._ensure_db_exists()
        
    def _ensure_db_exists(self):
        """Ensure the database file exists with proper structure."""
        if not self.db_path.exists():
            # Create directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize empty database
            initial_data = {
                "training_data": {}
            }
            self._save_data(initial_data)
            
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return empty structure if file doesn't exist or is corrupted
            return {
                "training_data": {}
            }
            
    def _save_data(self, data: Dict[str, Any]):
        """Save data to JSON file."""
        # Write to temporary file first, then rename (atomic operation)
        temp_path = self.db_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_path.replace(self.db_path)
        
    def insert_training_data(self, data: Dict[str, Any]) -> str:
        """
        Insert or update training data.
        
        Args:
            data: Dictionary containing training data with keys:
                  - id: unique identifier
                  - problem: problem text
                  - answer: correct answer
                  - round: training round (optional, default=1)
                  - image: image information (optional)
                  - experience: list of experience strings (optional)
                  
        Returns:
            The ID of the inserted training data
        """
        with self.lock:
            db_data = self._load_data()
            
            training_id = data['id']
            
            # Prepare training data entry
            training_entry = {
                "id": training_id,
                "problem": data['problem'],
                "answer": data['answer'],
                "round": data.get('round', 1),
                "experience": data.get('experience', []),
                "llm_answers_and_score": []
            }

            # Handle image data
            if 'image' in data:
                if isinstance(data['image'], list):
                    training_entry['image'] = data['image']
                else:
                    training_entry['image'] = [data['image']]
            else:
                training_entry['image'] = []

            # Handle prompt embedding if provided
            if 'prompt_embedding' in data:
                training_entry['prompt_embedding'] = data['prompt_embedding']
                
            # If entry already exists, preserve existing LLM responses and experience
            if training_id in db_data['training_data']:
                existing_entry = db_data['training_data'][training_id]
                training_entry['llm_answers_and_score'] = existing_entry.get('llm_answers_and_score', [])
                # Preserve existing experience if not provided in new data
                if 'experience' not in data and 'experience' in existing_entry:
                    training_entry['experience'] = existing_entry['experience']
                
            db_data['training_data'][training_id] = training_entry
            self._save_data(db_data)
            
            return training_id
            
    def add_llm_response(self, training_data_id: str, response: str, 
                        scores: Dict[str, float], reflexion: str = "") -> bool:
        """
        Add LLM response and scores to existing training data.
        
        Args:
            training_data_id: ID of the training data
            response: LLM response text
            scores: Dictionary of evaluation scores (accuracy, format, reason, length)
            reflexion: Reflection on the reasoning process
            
        Returns:
            True if successful, False if training data not found
        """
        with self.lock:
            db_data = self._load_data()
            
            if training_data_id not in db_data['training_data']:
                return False
                
            llm_response = {
                "ans": response,
                "reflexion": reflexion,
                "accuracy": scores.get('accuracy', 0.0),
                "format": scores.get('format', 0.0),
                "reason": scores.get('reason', 0.0),
                "length": scores.get('length', 0.0)
            }
                
            db_data['training_data'][training_data_id]['llm_answers_and_score'].append(llm_response)
            
            self._save_data(db_data)
            return True
            
    def get_training_data(self, training_data_id: str) -> Optional[Dict[str, Any]]:
        """
        Get training data by ID.
        
        Args:
            training_data_id: ID of the training data
            
        Returns:
            Training data dictionary or None if not found
        """
        with self.lock:
            db_data = self._load_data()
            return db_data['training_data'].get(training_data_id)
            
    def get_all_training_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all training data.
        
        Returns:
            Dictionary of all training data
        """
        with self.lock:
            db_data = self._load_data()
            return deepcopy(db_data['training_data'])
            
    def get_training_data_by_round(self, round_number: int) -> List[Dict[str, Any]]:
        """
        Get all training data for a specific round.
        
        Args:
            round_number: Round number to filter by
            
        Returns:
            List of training data dictionaries
        """
        with self.lock:
            db_data = self._load_data()
            return [
                data for data in db_data['training_data'].values() 
                if data.get('round') == round_number
            ]
            
    def update_round(self, training_data_id: str, new_round: int) -> bool:
        """
        Update the round number for training data.
        
        Args:
            training_data_id: ID of the training data
            new_round: New round number
            
        Returns:
            True if successful, False if training data not found
        """
        with self.lock:
            db_data = self._load_data()
            
            if training_data_id not in db_data['training_data']:
                return False
                
            db_data['training_data'][training_data_id]['round'] = new_round
            
            self._save_data(db_data)
            return True
            
    def add_experience(self, training_data_id: str, experience: str) -> bool:
        """
        Add an experience entry to existing training data.
        
        Args:
            training_data_id: ID of the training data
            experience: Experience string to add
            
        Returns:
            True if successful, False if training data not found
        """
        with self.lock:
            db_data = self._load_data()
            
            if training_data_id not in db_data['training_data']:
                return False
                
            if 'experience' not in db_data['training_data'][training_data_id]:
                db_data['training_data'][training_data_id]['experience'] = []
                
            db_data['training_data'][training_data_id]['experience'].append(experience)
            
            self._save_data(db_data)
            return True
            
    def remove_experience(self, training_data_id: str, experience_index: int) -> bool:
        """
        Remove an experience entry by index from training data.
        
        Args:
            training_data_id: ID of the training data
            experience_index: Index of the experience to remove (0-based)
            
        Returns:
            True if successful, False if training data or experience not found
        """
        with self.lock:
            db_data = self._load_data()
            
            if training_data_id not in db_data['training_data']:
                return False
                
            experience_list = db_data['training_data'][training_data_id].get('experience', [])
            
            if experience_index < 0 or experience_index >= len(experience_list):
                return False
                
            experience_list.pop(experience_index)
            
            self._save_data(db_data)
            return True
            
    def update_experience(self, training_data_id: str, experience_index: int, new_experience: str) -> bool:
        """
        Update an experience entry by index.
        
        Args:
            training_data_id: ID of the training data
            experience_index: Index of the experience to update (0-based)
            new_experience: New experience string
            
        Returns:
            True if successful, False if training data or experience not found
        """
        with self.lock:
            db_data = self._load_data()
            
            if training_data_id not in db_data['training_data']:
                return False
                
            experience_list = db_data['training_data'][training_data_id].get('experience', [])
            
            if experience_index < 0 or experience_index >= len(experience_list):
                return False
                
            experience_list[experience_index] = new_experience
            
            self._save_data(db_data)
            return True
            
    def get_experiences(self, training_data_id: str) -> Optional[List[str]]:
        """
        Get all experiences for a training data entry.
        
        Args:
            training_data_id: ID of the training data
            
        Returns:
            List of experience strings or None if training data not found
        """
        with self.lock:
            db_data = self._load_data()
            
            if training_data_id not in db_data['training_data']:
                return None
                
            return db_data['training_data'][training_data_id].get('experience', []).copy()
            
    def search_by_experience(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search training data by experience content.
        
        Args:
            search_term: Text to search for in experience entries
            
        Returns:
            List of matching training data
        """
        with self.lock:
            db_data = self._load_data()
            matching_data = []
            
            for data in db_data['training_data'].values():
                experiences = data.get('experience', [])
                for experience in experiences:
                    if search_term.lower() in experience.lower():
                        matching_data.append(data)
                        break  # Don't add the same data multiple times
                        
            return matching_data
            
    def delete_training_data(self, training_data_id: str) -> bool:
        """
        Delete training data by ID.
        
        Args:
            training_data_id: ID of the training data to delete
            
        Returns:
            True if successful, False if training data not found
        """
        with self.lock:
            db_data = self._load_data()
            
            if training_data_id not in db_data['training_data']:
                return False
                
            del db_data['training_data'][training_data_id]
            self._save_data(db_data)
            return True
            
    def search_by_problem_text(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search training data by problem text.
        
        Args:
            search_term: Text to search for in problem field
            
        Returns:
            List of matching training data
        """
        with self.lock:
            db_data = self._load_data()
            return [
                data for data in db_data['training_data'].values()
                if search_term.lower() in data.get('problem', '').lower()
            ]
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        with self.lock:
            db_data = self._load_data()
            training_data = db_data['training_data']
            
            total_entries = len(training_data)
            total_responses = sum(len(data.get('llm_answers_and_score', [])) for data in training_data.values())
            
            # Round distribution
            rounds = {}
            for data in training_data.values():
                round_num = data.get('round', 1)
                rounds[round_num] = rounds.get(round_num, 0) + 1
                
            # Score statistics
            all_scores = {'accuracy': [], 'format': [], 'reason': [], 'length': []}
            for data in training_data.values():
                for response in data.get('llm_answers_and_score', []):
                    for score_type in all_scores:
                        if score_type in response:
                            all_scores[score_type].append(response[score_type])
                            
            avg_scores = {}
            for score_type, scores in all_scores.items():
                if scores:
                    avg_scores[f'avg_{score_type}'] = sum(scores) / len(scores)
                else:
                    avg_scores[f'avg_{score_type}'] = 0.0
                    
            # Experience statistics
            total_experiences = sum(len(data.get('experience', [])) for data in training_data.values())
            entries_with_experience = sum(1 for data in training_data.values() if data.get('experience', []))
                    
            return {
                'total_training_entries': total_entries,
                'total_llm_responses': total_responses,
                'total_experiences': total_experiences,
                'entries_with_experience': entries_with_experience,
                'round_distribution': rounds,
                'average_scores': avg_scores
            }
            
    def get_llm_responses_sorted_by_average_score(self, training_data_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get LLM responses for a training data ID, sorted by average score.
        
        Args:
            training_data_id: ID of the training data
            
        Returns:
            List of LLM responses sorted by average score (highest first), or None if training data not found
        """
        with self.lock:
            db_data = self._load_data()
            
            if training_data_id not in db_data['training_data']:
                return None
                
            llm_responses = db_data['training_data'][training_data_id].get('llm_answers_and_score', [])
            
            if not llm_responses:
                return []
            
            # Calculate average score for each response and sort
            def calculate_average_score(response: Dict[str, Any]) -> float:
                """Calculate average of accuracy, format, reason, and length scores."""
                score_types = ['accuracy', 'format', 'reason', 'length']
                scores = []
                for score_type in score_types:
                    if score_type in response:
                        scores.append(float(response[score_type]))
                    else:
                        scores.append(0.0)  # Default to 0 if score is missing
                
                return sum(scores) / len(scores) if scores else 0.0
            
            # Create a copy of responses with average scores and sort
            responses_with_avg = []
            for response in llm_responses:
                response_copy = deepcopy(response)
                response_copy['average_score'] = calculate_average_score(response)
                responses_with_avg.append(response_copy)
            
            # Sort by average score in descending order (highest first)
            sorted_responses = sorted(responses_with_avg, key=lambda x: x['average_score'], reverse=True)
            
            return sorted_responses

    def change_reflexion(self, training_data_id: str, response_index: int, new_reflexion: str) -> bool:
        """
        Update the reflexion for a specific LLM response.
        
        Args:
            training_data_id: ID of the training data
            response_index: Index of the LLM response to update (0-based)
            new_reflexion: New reflexion text
            
        Returns:
            True if successful, False if training data or response not found
        """
        with self.lock:
            db_data = self._load_data()
            
            # Check if training data exists
            if training_data_id not in db_data['training_data']:
                return False
                
            llm_responses = db_data['training_data'][training_data_id].get('llm_answers_and_score', [])
            
            # Check if response index is valid
            if response_index < 0 or response_index >= len(llm_responses):
                return False
                
            # Update the reflexion
            llm_responses[response_index]['reflexion'] = new_reflexion
            
            self._save_data(db_data)
            return True
            
    def change_reflexion_by_response_content(self, training_data_id: str, response_text: str, new_reflexion: str) -> bool:
        """
        Update the reflexion for an LLM response by matching response content.
        
        Args:
            training_data_id: ID of the training data
            response_text: Text content of the response to find (partial match)
            new_reflexion: New reflexion text
            
        Returns:
            True if successful, False if training data or response not found
        """
        with self.lock:
            db_data = self._load_data()
            
            # Check if training data exists
            if training_data_id not in db_data['training_data']:
                return False
                
            llm_responses = db_data['training_data'][training_data_id].get('llm_answers_and_score', [])
            
            # Find response by content
            for response in llm_responses:
                if response_text.lower() in response.get('ans', '').lower():
                    response['reflexion'] = new_reflexion
                    self._save_data(db_data)
                    return True
                    
            return False

    def export_to_json(self, output_path: str, round_filter: Optional[int] = None) -> bool:
        """
        Export data to a JSON file.
        
        Args:
            output_path: Path for the output file
            round_filter: Optional round number to filter by
            
        Returns:
            True if successful
        """
        with self.lock:
            db_data = self._load_data()
            
            if round_filter is not None:
                filtered_data = {
                    k: v for k, v in db_data['training_data'].items()
                    if v.get('round') == round_filter
                }
                export_data = {
                    'training_data': filtered_data
                }
            else:
                export_data = db_data
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            return True


# Example usage and utility functions
def create_sample_data():
    """Create sample data for testing."""
    return {
        'id': 'trance-001',
        'problem': 'What is the spatial transformation applied to the object?',
        'answer': 'rotation by 90 degrees clockwise',
        'round': 1,
        'experience': [
            'First attempt: Identified rotation direction correctly',
            'Learned to focus on spatial relationships between objects'
        ],
        'image': [
            {
                'path': 'Spatial-Transformation/image1.png',
                'type': 'Spatial-Transformation',
                'transformation': 'rotation',
                'angle': 90
            }
        ]
    }


def create_sample_llm_response():
    """Create sample LLM response data."""
    return {
        'response': 'The object appears to be rotated 90 degrees clockwise based on visual analysis.',
        'reflexion': 'I analyzed the spatial relationship between the source and target images to identify the transformation.',
        'scores': {
            'accuracy': 0.85,
            'format': 0.90,
            'reason': 0.75,
            'length': 0.80
        }
    }


if __name__ == "__main__":
    # Example usage
    db = JSONDatabase("test_memory_db.json")
    
    # Insert sample data
    sample_data = create_sample_data()
    training_id = db.insert_training_data(sample_data)
    print(f"Inserted training data with ID: {training_id}")
    
    # Add multiple sample LLM responses with different scores
    sample_responses = [
        {
            'response': 'The object appears to be rotated 90 degrees clockwise based on visual analysis.',
            'scores': {'accuracy': 0.85, 'format': 0.90, 'reason': 0.75, 'length': 0.80}
        },
        {
            'response': 'Looking at the transformation, it seems like a 90-degree rotation.',
            'scores': {'accuracy': 0.95, 'format': 0.60, 'reason': 0.85, 'length': 0.70}
        },
        {
            'response': 'The spatial transformation is a clockwise rotation of 90 degrees.',
            'scores': {'accuracy': 0.90, 'format': 0.95, 'reason': 0.80, 'length': 0.85}
        }
    ]
    
    for i, sample_response in enumerate(sample_responses):
        success = db.add_llm_response(
            training_id, 
            sample_response['response'], 
            sample_response['scores'],
            reflexion=""
        )
        print(f"Added LLM response {i+1}: {success}")
    
    # Test the new function - get responses sorted by average score
    sorted_responses = db.get_llm_responses_sorted_by_average_score(training_id)
    if sorted_responses:
        print(f"\nLLM responses sorted by average score (highest first):")
        for i, response in enumerate(sorted_responses):
            avg_score = response.get('average_score', 0.0)
            print(f"  Response {i+1}: Average score = {avg_score:.3f}")
            print(f"    Accuracy: {response.get('accuracy', 0.0):.2f}, "
                  f"Format: {response.get('format', 0.0):.2f}, "
                  f"Reason: {response.get('reason', 0.0):.2f}, "
                  f"Length: {response.get('length', 0.0):.2f}")
            print(f"    Response: {response.get('ans', '')[:50]}...")
    
    # Test the change_reflexion functions
    print(f"\nTesting reflexion change functions:")
    
    # Change reflexion by index
    success = db.change_reflexion(training_id, 0, "Updated: I carefully analyzed the spatial relationships and transformations.")
    print(f"Changed reflexion for response 0: {success}")
    
    # Change reflexion by response content
    success = db.change_reflexion_by_response_content(
        training_id, 
        "90-degree rotation", 
        "Modified: I used geometric analysis to determine the transformation type and angle."
    )
    print(f"Changed reflexion by content match: {success}")
    
    # Verify the changes
    updated_data = db.get_training_data(training_id)
    if updated_data:
        print(f"\nUpdated reflexions:")
        for i, response in enumerate(updated_data.get('llm_answers_and_score', [])):
            print(f"  Response {i+1} reflexion: {response.get('reflexion', 'No reflexion')}")
    else:
        print("Failed to retrieve updated data")
    
    # Test experience functions
    print(f"\nTesting experience functions:")
    
    # Add experiences
    success = db.add_experience(training_id, "Learning: Pay attention to angle measurements")
    print(f"Added experience 1: {success}")
    
    success = db.add_experience(training_id, "Insight: Visual comparison helps identify transformations")
    print(f"Added experience 2: {success}")
    
    # Get all experiences
    experiences = db.get_experiences(training_id)
    print(f"All experiences: {experiences}")
    
    # Update an experience
    success = db.update_experience(training_id, 1, "Updated insight: Visual and geometric analysis combined")
    print(f"Updated experience at index 1: {success}")
    
    # Search by experience
    matching_data = db.search_by_experience("visual")
    print(f"Found {len(matching_data)} entries with 'visual' in experience")
    
    # Remove an experience
    success = db.remove_experience(training_id, 0)
    print(f"Removed experience at index 0: {success}")
    
    # Get experiences after removal
    experiences = db.get_experiences(training_id)
    print(f"Experiences after removal: {experiences}")
    
    # Retrieve and display data
    result = db.get_training_data(training_id)
    print(f"\nRetrieved data:")
    print(json.dumps(result, indent=2))
    
    # Show statistics
    stats = db.get_statistics()
    print(f"\nDatabase statistics:")
    print(json.dumps(stats, indent=2))
