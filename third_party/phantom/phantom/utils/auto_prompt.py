import os
import argparse
import json
from pathlib import Path
from dashscope import MultiModalConversation
import dashscope
from tqdm import tqdm


# Fixed English prompt template
FIXED_PROMPT_TEMPLATE = """Briefly describe the scene in the image. Focus on describing the table and objects on the table,
as well as the spatial relationships between objects and the relative positions of objects to the human hand.
Briefly describe the initial state of the human hand. Describe in detail the task, the entire action sequence,
the object's appearance, and the physical interactions between the human hand and the object. Briefly describe the final position state of the human hand and the manipulated object."""


def generate_prompt_with_qwen(
    image_path: str,
    task_description: str,
    model: str = "qwen3-vl-flash",
    api_key: str = None,
    base_url: str = None
):
    """
    Generate English prompt based on image and task description using Qwen VLM model
    
    Args:
        image_path: Absolute path to the image
        task_description: Task description (e.g., "Robotic arm picks up the banana on the table")
        model: Model name
        api_key: API key
        base_url: API base URL
    
    Returns:
        Generated English prompt
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    
    # Set API base URL (if specified)
    if base_url:
        dashscope.base_http_api_url = base_url
    else:
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    
    # Image path needs to add file:// prefix
    image_path_with_prefix = f"file://{image_path}"
    
    # Build input prompt (using fixed English template, task description can be any language)
    query_text = f"""Please analyze this image and generate a detailed English prompt following the template below.

Template requirements:
{FIXED_PROMPT_TEMPLATE}

Task description: {task_description}

You cannot use punctuation other than periods, and don't make the prompt too long.

Generate only the prompt text in English, no additional explanations."""
    
    # Build messages
    messages = [
        {
            'role': 'user',
            'content': [
                {'image': image_path_with_prefix},
                {'text': query_text}
            ]
        }
    ]
    
    # Get API key
    if api_key is None:
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if api_key is None:
            raise ValueError(
                "API key not found. Please set the DASHSCOPE_API_KEY environment variable or provide it via the --api_key parameter.\n"
                "To get an API Key: https://help.aliyun.com/zh/model-studio/get-api-key"
            )
    
    # Call API
    try:
        response = MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages
        )
        
        # Check if the response is successful
        if response.status_code != 200:
            error_msg = f"API call failed: {response.message} (Status code: {response.status_code})"
            if "Model not exist" in str(response.message) or response.status_code == 400:
                error_msg += f"\nHint: Model '{model}' may not exist or the name is incorrect."
                error_msg += "\nPlease try using one of the following models:"
                error_msg += "\n  - qwen-vl-plus (Recommended)"
                error_msg += "\n  - qwen-vl-max"
                error_msg += "\n  - qwen-vl-7b-chat"
                error_msg += "\n  - qwen2-vl-7b-instruct"
                error_msg += "\nMore models: https://help.aliyun.com/zh/model-studio/models"
            raise RuntimeError(error_msg)
        
        # Extract generated text
        if hasattr(response, 'output') and hasattr(response.output, 'choices'):
            if len(response.output.choices) > 0:
                message = response.output.choices[0].message
                if hasattr(message, 'content') and len(message.content) > 0:
                    # content might be a list containing multiple elements (text, images, etc.)
                    text_content = None
                    for item in message.content:
                        if isinstance(item, dict) and "text" in item:
                            text_content = item["text"]
                            break
                        elif isinstance(item, str):
                            text_content = item
                            break
                    
                    if text_content:
                        return text_content.strip()
        
        # Try direct access if extraction from standard location fails
        if hasattr(response, 'output'):
            output_str = str(response.output)
            if output_str:
                return output_str.strip()
        
        raise RuntimeError("Failed to extract text content from API response")
        
    except Exception as e:
        raise RuntimeError(f"Error calling Qwen API: {str(e)}")


def process_single_image(
    image_path: Path,
    input_folder: Path,
    task_description: str,
    model: str,
    api_key: str,
    base_url: str,
    num_output_frames: int = 93,
    chunk_size: int = 93,
    chunk_overlap: int = 1
):
    """
    Process a single image, generate prompt and json files
    
    Args:
        image_path: Image path
        input_folder: Input folder path
        task_description: Task description
        model: Model name
        api_key: API key
        base_url: API base URL
        num_output_frames: Number of output frames (for json)
        chunk_size: Chunk size (for json)
        chunk_overlap: Chunk overlap (for json)
    """
    image_name = image_path.stem  # Filename without extension
    
    print(f"\nProcessing image: {image_path.name}")
    
    # Generate prompt
    try:
        generated_prompt = generate_prompt_with_qwen(
            str(image_path),
            task_description,
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        
        # Save txt file
        txt_path = input_folder / f"{image_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(generated_prompt)
        print(f"  ✓ Prompt saved to: {txt_path.name}")
        
        # Generate and save json file
        json_data = {
            "inference_type": "image2world",
            "name": image_name,
            "input_path": f"{image_name}.png",
            "prompt_path": f"{image_name}.txt",
            "num_output_frames": num_output_frames,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        json_path = input_folder / f"{image_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"  ✓ JSON saved to: {json_path.name}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Processing failed: {e}")
        return False


def batch_process_images(
    input_folder: str,
    task_description: str,
    model: str = "qwen3-vl-flash",
    api_key: str = None,
    base_url: str = None,
    num_output_frames: int = 93,
    chunk_size: int = 93,
    chunk_overlap: int = 1
):
    """
    Batch process all images in the folder
    
    Args:
        input_folder: Input folder path
        task_description: Task description
        model: Model name
        api_key: API key
        base_url: API base URL
        num_output_frames: Number of output frames (for json)
        chunk_size: Chunk size (for json)
        chunk_overlap: Chunk overlap (for json)
    """
    input_folder = Path(input_folder)
    
    if not input_folder.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    if not input_folder.is_dir():
        raise ValueError(f"Input path is not a folder: {input_folder}")
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_folder.glob(f"*{ext}")))
    
    if len(image_files) == 0:
        print(f"No image files found in folder {input_folder}")
        return
    
    print(f"Found {len(image_files)} image files")
    print(f"Task description: {task_description}")
    print(f"Using model: {model}")
    
    # Process each image
    success_count = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        if process_single_image(
            image_path,
            input_folder,
            task_description,
            model,
            api_key,
            base_url,
            num_output_frames,
            chunk_size,
            chunk_overlap
        ):
            success_count += 1
    
    print(f"\nProcessing complete! Successfully processed {success_count}/{len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images, generating English prompts and JSON config files using the Qwen VLM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch process all images in a folder
  python %(prog)s /path/to/input/folder "Robotic arm picks up the banana on the table"
  
  # Specify model and API key
  python %(prog)s /path/to/input/folder "Robotic arm picks up the banana on the table" \\
      --model qwen-vl-plus --api_key sk-xxx
  
  # Custom JSON parameters
  python %(prog)s /path/to/input/folder "Robotic arm picks up the banana on the table" \\
      --num_output_frames 100 --chunk_size 100 --chunk_overlap 2

Supported models:
  - qwen3-vl-flash (Default)
  - qwen-vl-plus
  - qwen-vl-max
  - qwen-vl-7b-chat
  - qwen2-vl-7b-instruct
  More models: https://help.aliyun.com/zh/model-studio/models

To get an API Key: https://help.aliyun.com/zh/model-studio/get-api-key
        """
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Input folder path (containing image files)"
    )
    parser.add_argument(
        "task_description",
        type=str,
        help="Task description (e.g., 'Robotic arm picks up the banana on the table')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-flash",
        help="Qwen model name (Default: qwen3-vl-flash)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default='',
        help="DashScope API key (If not provided, it will be retrieved from the DASHSCOPE_API_KEY environment variable)"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="API base URL (Default: Beijing region. Optional: Virginia region https://dashscope-us.aliyuncs.com/api/v1, Singapore region https://dashscope-intl.aliyuncs.com/api/v1)"
    )
    parser.add_argument(
        "--num_output_frames",
        type=int,
        default=93,
        help="num_output_frames in JSON file (Default: 93)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=93,
        help="chunk_size in JSON file (Default: 93)"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=1,
        help="chunk_overlap in JSON file (Default: 1)"
    )
    
    args = parser.parse_args()
    
    # Batch process images
    try:
        batch_process_images(
            args.input_folder,
            args.task_description,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            num_output_frames=args.num_output_frames,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())