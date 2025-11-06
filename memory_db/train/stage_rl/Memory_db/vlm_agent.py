import json, os
from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


import argparse

# ====================================
#  COT PROMPT
# ====================================

COT_TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.  

The **attributes of the initial objects** are provxided as a list of tuples in the following format:  
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

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
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
   
### Output Format  

You should first thinks about the reasoning process internally and then provides the user with the answer. The **reasoning process** and **answer** are enclosed within specific tags:  

- **Reasoning process**: Enclosed within `<think>...</think>`  
- **Final answer (sequence of functions only)**: Enclosed within `<answer>...</answer>`  

Now, it's your turn!

{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
'''

COT_CLEVR_MATH_QUESTION_PROMPT = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

COT_GEOMATH_CHOICE_QUESTION_PROMPT = "{Question} Please select the correct answer by writing the letter (A, B, C or D) that precedes your choice.\nOutput the thinking process in <think> </think> and final answer (chosen letter) in <answer> </answer> tags."

COT_GEOMATH_NON_CHOICE_QUESTION_PROMPT = "{Question} Output the thinking process in <think> </think> and final answer (float number or int number) in <answer> </answer> tags."

COT_GEOMETRY_QUESTION_PROMPT = "{Question} Output the thinking process in <think> </think> and final answer (number or choice) in <answer> </answer> tags."

COT_TRANCE_QUESTION_WITH_CAPTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

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

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
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
   
### Output Format  

You should first thinks about the reasoning process internally and then provides the user with the answer. The **reasoning process** and **answer** are enclosed within specific tags:  

- **Summary process**: Summary how you will approach the problem and explain the steps you will take to reach the answer, enclosed within `<summary>...</summary>`

- **Caption process**: Provide a detailed description of the image, particularly emphasizing the aspects related to the question, enclosed within `<caption>...</caption>`

- **Reasoning process**: Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning, enclosed within `<think>...</think>`  

- **Final answer (sequence of functions only)**: Enclosed within `<answer>...</answer>`

Now, it's your turn!

{Question} Output the summary process in <summary> </summary>, caption process in <caption>...</caption>, thinking process in <think> </think> and final answer in <answer> </answer> tags.
'''

# ====================================
#  SFT PROMPT
# ====================================

SFT_TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

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

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
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

Now, it's your turn!

{Question}
'''

SFT_CLEVR_MATH_QUESTION_PROMPT = "{Question}"

SFT_GEOMATH_QUESTION_PROMPT = "{Question}"

SFT_GEOMETRY_QUESTION_PROMPT = "{Question}"

# ====================================
#  Zero-Shot PROMPT
# ====================================

ZERO_SHOT_TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

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

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
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

Now, it's your turn!

{Question} Please output the answer only with a sequence of functions for transformation.
'''

ZERO_SHOT_CLEVR_MATH_QUESTION_PROMPT = "Please answer in Arabic numerals. For example, if the answer is 3, please respond with 3. {Question}"

ZERO_SHOT_GEOMATH_QUESTION_PROMPT = "Please answer the question with only numbers (either integer or float, such as 1, 2, 5.2, etc.) or options (such as A, B, C, or D). If it is an option, please provide your answer as a single letter (A, B, C, or D). For example, if the answer is A, just respond with A. Do not include any explanations or additional text. {Question}"


class VL_Evaluator():
    def __init__(self, model_name_or_path, max_image_num=2):
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
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path

    def generate_single(self, sample, image_dir, prompt_template, task_name="trance"):
        """Generate response for a single sample"""
        images = []
        
        # Process images
        if isinstance(sample["image"], list):
            for img_path in sample["image"]:
                image_full_path = os.path.join(image_dir, img_path)
                if os.path.exists(image_full_path):
                    images.append(Image.open(image_full_path))
        else:
            image_full_path = os.path.join(image_dir, sample["image"])
            if os.path.exists(image_full_path):
                images.append(Image.open(image_full_path))

        # Format the prompt
        question_text = prompt_template.format(Question=sample["question"])
        
        # Create conversation messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_text}
                ] + [{"type": "image", "image": img} for img in images]
            }
        ]

        # Apply chat template
        vllm_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate response
        prompt_with_images = {
            "prompt": vllm_prompt, 
            "multi_modal_data": {"image": images}
        }
        
        outputs = self.model.generate([prompt_with_images], sampling_params=self.sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        
        return generated_text

    def generate_batch(self, sample_list, image_dir, prompt_template, task_name="trance"):
        """Generate responses for a batch of samples"""
        prompts_text_and_vision = []
        
        for sample in sample_list:
            images = []
            
            # Process images
            if isinstance(sample["image"], list):
                for img_path in sample["image"]:
                    image_full_path = os.path.join(image_dir, img_path)
                    if os.path.exists(image_full_path):
                        images.append(Image.open(image_full_path))
            else:
                image_full_path = os.path.join(image_dir, sample["image"])
                if os.path.exists(image_full_path):
                    images.append(Image.open(image_full_path))

            # Format the prompt
            question_text = prompt_template.format(Question=sample["question"])
            
            # Create conversation messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_text}
                    ] + [{"type": "image", "image": img} for img in images]
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
                    "multi_modal_data": {"image": images}
                }
            )

        # Generate responses for the batch
        outputs = self.model.generate(prompts_text_and_vision, sampling_params=self.sampling_params, use_tqdm=False)

        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)

        return results


class QWEN_VL_Evaluator(VL_Evaluator):
    def __init__(self, model_name_or_path, max_image_num=2, min_pixels=3136, max_pixels=480000):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        try:
            self.processor.image_processor.min_pixels = min_pixels
            self.processor.image_processor.max_pixels = max_pixels
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
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path


class Mllama_VL_Evaluator(VL_Evaluator):
    def __init__(self, model_name_or_path, max_image_num=2):
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
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path


class PHI3V_VL_Evaluator(VL_Evaluator):
    def __init__(self, model_name_or_path, max_image_num=2):
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
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path


class Pixtral_VL_Evaluator(VL_Evaluator):
    def __init__(self, model_name_or_path, max_image_num=2):
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
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path


class Internvl_VL_Evaluator(VL_Evaluator):
    def __init__(self, model_name_or_path, max_image_num=2):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
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
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path


def create_vlm_agent(model_name_or_path, max_image_num=2):
    """Factory function to create the appropriate VLM evaluator based on model name"""
    print(f"Loading Model from {model_name_or_path} ...")
    
    if 'qwen' in model_name_or_path.lower():
        print("======== Using QWEN_VL_Evaluator ==========")
        return QWEN_VL_Evaluator(model_name_or_path, max_image_num)
    elif 'llama' in model_name_or_path.lower() and 'vision' in model_name_or_path.lower():
        print("======== Using Mllama_VL_Evaluator ==========")
        return Mllama_VL_Evaluator(model_name_or_path, max_image_num)
    elif 'phi' in model_name_or_path.lower():
        print("======== Using PHI3V_VL_Evaluator ==========")
        return PHI3V_VL_Evaluator(model_name_or_path, max_image_num)
    elif 'internvl' in model_name_or_path.lower():
        print("======== Using Internvl_VL_Evaluator ==========")
        return Internvl_VL_Evaluator(model_name_or_path, max_image_num)
    elif 'pixtral' in model_name_or_path.lower():
        print("======== Using Pixtral_VL_Evaluator ==========")
        return Pixtral_VL_Evaluator(model_name_or_path, max_image_num)
    else:
        print("======== Using Default VL_Evaluator ==========")
        return VL_Evaluator(model_name_or_path, max_image_num)


def get_prompt_template(task_name, eval_type):
    """Get the appropriate prompt template based on task and evaluation type"""
    if eval_type == "cot-sft":
        if task_name in ["trance", "trance-left", "trance-right"]:
            return COT_TRANCE_QUESTION_PROMPT
        elif task_name in ["clevr-math", "super-clevr"]:
            return COT_CLEVR_MATH_QUESTION_PROMPT
        elif task_name in ["geomath"]:
            return COT_GEOMATH_CHOICE_QUESTION_PROMPT
        elif task_name in ["geometry3k"]:
            return COT_GEOMETRY_QUESTION_PROMPT
    elif eval_type == "sft":
        if task_name in ["trance", "trance-left", "trance-right"]:
            return SFT_TRANCE_QUESTION_PROMPT
        elif task_name in ["clevr-math"]:
            return SFT_CLEVR_MATH_QUESTION_PROMPT
        elif task_name in ["geomath"]:
            return SFT_GEOMATH_QUESTION_PROMPT
        elif task_name in ["geometry3k"]:
            return SFT_GEOMETRY_QUESTION_PROMPT
    elif eval_type == "zero-shot":
        if task_name in ["trance", "trance-left", "trance-right"]:
            return ZERO_SHOT_TRANCE_QUESTION_PROMPT
        elif task_name in ["clevr-math", "super-clevr"]:
            return ZERO_SHOT_CLEVR_MATH_QUESTION_PROMPT
        elif task_name in ["geomath", "geometry3k"]:
            return ZERO_SHOT_GEOMATH_QUESTION_PROMPT
    elif eval_type == "caption-cot":
        return COT_TRANCE_QUESTION_WITH_CAPTION_PROMPT
    
    # Default fallback
    return COT_TRANCE_QUESTION_PROMPT


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Agent for visual question answering.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--task_name', type=str, default="trance", help="Task name (default: trance)")
    parser.add_argument('--eval_type', type=str, default="cot-sft", help="Evaluation type (default: cot-sft)")
    parser.add_argument('--max_image_num', type=int, default=2, help="Maximum number of images (default: 2)")
    
    args = parser.parse_args()
    
    # Create VLM agent
    vlm_agent = create_vlm_agent(args.model_name_or_path, args.max_image_num)
    
    # Get prompt template
    prompt_template = get_prompt_template(args.task_name, args.eval_type)
    
    print(f"VLM Agent created successfully!")
    print(f"Model: {args.model_name_or_path}")
    print(f"Task: {args.task_name}")
    print(f"Evaluation Type: {args.eval_type}")
    
    # Example of how to use the agent
    # sample = {
    #     "image": ["image1.jpg", "image2.jpg"],  # or single image "image.jpg"
    #     "question": "Your question here"
    # }
    # image_dir = "/path/to/images"
    # response = vlm_agent.generate_single(sample, image_dir, prompt_template, args.task_name)
    # print(f"Response: {response}")