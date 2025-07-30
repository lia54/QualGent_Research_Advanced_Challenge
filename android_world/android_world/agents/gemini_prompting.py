# gemini_prompting.py

import base64
import os
import io
from PIL import Image
#from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part, Image as VertexAIImage, GenerationConfig
#from google.generativeai import types

import PIL.Image
import google.generativeai as genai
from tqdm import tqdm
import subprocess
from datetime import datetime # Import datetime for timestamps
import json # Import json for saving results

# --- Configuration ---
# Replace with your Google Cloud Project ID and Location
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project-466322")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1") # Or your preferred region
GEMINI_MODEL_NAME = "gemini-2.5-pro" # Or "gemini-2.5-flash" for potentially faster/cheaper inference

RESULTS_DIR = "results" # Define the results directory name, consistent with main script

# Assuming COLOR_BLUE, COLOR_YELLOW, COLOR_MAGENTA, COLOR_RESET are defined elsewhere
COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"
COLOR_GREEN = "\033[32m"
COLOR_CYAN = "\033[36m"
COLOR_RED = "\033[31m"
COLOR_RESET = "\033[0m"

# --- Helper Function for Vision-to-Prompt Generation ---

def generate_prompt_from_screenshot(
    image_path: str,
    task_id: str, # Added task_id for naming result files
    task_goal: str,
    description_prompt: str = None,
    few_shot_examples: list = None
) -> str:
    """
    Analyzes an Android UI screenshot using Gemini 2.5 Pro Vision and generates a text prompt
    describing the UI elements and suggesting actions to achieve the task goal.

    Args:
        image_path (str): Path to the screenshot image file.
        task_goal (str): The high-level goal the agent needs to achieve (e.g., "Navigate to settings").
        description_prompt (str, optional): A specific prompt to guide Gemini's description.
        few_shot_examples (list, optional): List of dictionaries, each containing 'image' (base64 str),
                                           'text_description', and 'action_suggestion'.

    Returns:
        str: The generated text prompt for the LLM agent.
    """
    if not os.path.exists(image_path):
        tqdm.write(f"{COLOR_YELLOW}Error: Image file not found at {image_path}{COLOR_RESET}")
        return f"Error: Image file not found at {image_path}"

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tqdm.write(f"{COLOR_BLUE}Analyzing screenshot '{image_path}' with Gemini 2.5 Pro Vision...{COLOR_RESET}")

    api_key = os.environ.get("GCP_API_KEY")
    if not api_key:
        raise ValueError("Google API Key not found in environment variables. Please set GOOGLE_API_KEY.")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-1.5-pro-latest') 

    # Load and encode the image
    try:
        # Load the image
        img = PIL.Image.open(image_path) # Replace with your image path

    except Exception as e:
        tqdm.write(f"{COLOR_YELLOW}Error loading image {image_path}: {e}{COLOR_RESET}")
        return f"Error loading image {image_path}: {e}"

    # Construct the multimodal prompt for Gemini
    # Guide Gemini to extract UI elements and suggest actions
    system_instruction = (
        "You are an expert Android UI analyst. Your task is to analyze the provided screenshot "
        "and generate a clear, concise text description of the UI elements present. "
        "Then, based on the task goal, you should suggest a plan or sequence of high-level actions "
        "an automated agent could take to achieve that goal. Be specific about UI elements "
        "identified (e.g., 'Click button with text X', 'Tap icon Y', 'Input text Z')."
    )

    if description_prompt:
        system_instruction += f"\nSpecific focus: {description_prompt}"

    current_prompt_text = f"\nNow, for the current screen, given the task: '{task_goal}', describe the UI and suggest the next actions."

    # Create the content list with text and image
    content = [system_instruction + current_prompt_text, img]
    
    # Generation configuration for Gemini
    # not used so it get default values
    # generation_config = GenerationConfig(
    #     temperature=0.2, # Lower temperature for more deterministic/factual descriptions
    #     max_output_tokens=1024
    # )
    
    result_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "task_id": task_id,
        "task_goal": task_goal,
        "image_path": image_path,
        "description_prompt_input": description_prompt,
        "few_shot_examples_input": few_shot_examples,
        "gemini_output": None,
        "generated_prompt_for_llm": None,
        "error": None
    }

    try:
        # Send the content (text and image)
        response = model.generate_content(content)

        gemini_output = response.candidates[0].content.parts[0].text
        #tqdm.write(f"{COLOR_BLUE}Gemini 2.5 Output:{COLOR_RESET}\n{gemini_output}\n")

        # You might refine the Gemini output here to directly form the prompt
        # for your LLM agent. For example, extract action suggestions.
        # This initial version simply takes Gemini's full text.
        generated_prompt = f"Current UI Description and Suggested Actions:\n{gemini_output}\n\nTask Goal: {task_goal}\n\nBased on the above, prepare to provide the optimal action as before."

        result_data["gemini_output"] = gemini_output
        result_data["generated_prompt_for_llm"] = generated_prompt
        
        #tqdm.write(f"{COLOR_BLUE}Generated Prompt for LLM Agent:{COLOR_RESET}\n{generated_prompt}\n")
        
        # Save the results to a JSON file
        safe_task_id = task_id.replace(" ", "_").replace("/", "__").lower()
        filename = f"gemini_prompt_{safe_task_id}_{result_data['timestamp']}.json"
        filepath = os.path.join(RESULTS_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(result_data, f, indent=4)
        tqdm.write(f"{COLOR_BLUE}Gemini prompt results saved to {filepath}{COLOR_RESET}")

        
        return generated_prompt

    except Exception as e:
        tqdm.write(f"{COLOR_YELLOW}Error calling Gemini API: {e}{COLOR_RESET}")
        return f"Error calling Gemini API: {e}"


# --- Example Usage (when run as a script) ---

if __name__ == "__main__":
    # --- Example 1: Basic Screenshot Analysis ---
    example_image_path_1 = "Screenshot_1753561726.png" # Replace with a real screenshot file
    example_task_id_1 = "task_nav_settings_001" # Added task_id
    example_task_goal_1 = "Open the settings menu and navigate to display options."

    # Create a dummy image for demonstration if it doesn't exist
    if not os.path.exists(example_image_path_1):
        tqdm.write(f"{COLOR_BLUE}Creating a dummy image for example: {example_image_path_1}{COLOR_RESET}")
        dummy_image = Image.new('RGB', (100, 100), color = 'red')
        dummy_image.save(example_image_path_1)

    generated_prompt_1 = generate_prompt_from_screenshot(
        image_path=example_image_path_1,
        task_id=example_task_id_1, # Pass task_id
        task_goal=example_task_goal_1
    )
    tqdm.write(f"\n--- Example 1 Prompt Generated ---")
    tqdm.write(generated_prompt_1)
    tqdm.write("--------------------------------\n")

    # --- Example 2: With specific description prompt and (dummy) few-shot examples ---
    example_image_path_2 = "Screenshot_1753561472.png" # Replace with another real screenshot file
    example_task_id_2 = "task_add_item_001" # Added task_id
    example_task_goal_2 = "Click the 'Add New Item' button."
    specific_description_prompt = "Identify buttons, text fields, and icons clearly."

    # Create another dummy image for demonstration if it doesn't exist
    if not os.path.exists(example_image_path_2):
        tqdm.write(f"{COLOR_BLUE}Creating a dummy image for example: {example_image_path_2}{COLOR_RESET}")
        dummy_image_2 = Image.new('RGB', (100, 100), color = 'blue')
        dummy_image_2.save(example_image_path_2)

    # Dummy few-shot examples (real examples would be images + desired descriptions/actions)
    # The 'image' field would be a base64 encoded string of an image file
    with open(example_image_path_1, "rb") as image_file:
        encoded_image_1 = base64.b64encode(image_file.read()).decode("utf-8")

    few_shot_examples_data_2 = [
        {
            "image": encoded_image_1,
            "text_description": "The screen shows a list of contacts. There is an 'Add Contact' button at the bottom.",
            "action_suggestion": "Plan: Identify the 'Add Contact' button. Action: click(id:add_contact_button)"
        }
    ]

    generated_prompt_2 = generate_prompt_from_screenshot(
        image_path=example_image_path_2,
        task_id=example_task_id_2, # Pass task_id
        task_goal=example_task_goal_2,
        description_prompt=specific_description_prompt,
        few_shot_examples=few_shot_examples_data_2
    )
    tqdm.write(f"\n--- Example 2 Prompt Generated ---")
    tqdm.write(generated_prompt_2)
    tqdm.write("--------------------------------\n")
    
    # --- Example 3: With specific description prompt, adb screenshot and few-shot examples ---
    def take_screenshot(device_id=None, output_file="screenshot.png"):
        try:
            # Construct the ADB command
            adb_command = ["adb"]
            if device_id:
                adb_command.extend(["-s", device_id])  # Specify device if multiple are connected
            adb_command.extend(["exec-out", "screencap", "-p"])

            # Run the ADB command and capture the screenshot
            with open(output_file, "wb") as file:
                subprocess.run(adb_command, stdout=file, check=True)
            
            print(f"Screenshot saved as {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error taking screenshot: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
    take_screenshot(output_file="example_screenshot_3.png") # Replace with another real screenshot file
    example_image_path_3 = "example_screenshot_3.png"
    example_task_id_3 = "task_click_gmail_001" # Added task_id
    example_task_goal_3 = "Click the 'Add New Item' button."
    specific_description_prompt = "Identify buttons, text fields, and icons clearly."

    # Create another dummy image for demonstration if it doesn't exist
    if not os.path.exists(example_image_path_3):
        tqdm.write(f"{COLOR_BLUE}Creating a dummy image for example: {example_image_path_3}{COLOR_RESET}")
        dummy_image_3 = Image.new('RGB', (100, 100), color = 'blue')
        dummy_image_3.save(example_image_path_3)

    # Dummy few-shot examples (real examples would be images + desired descriptions/actions)
    # The 'image' field would be a base64 encoded string of an image file
    with open(example_image_path_3, "rb") as image_file:
        encoded_image_3 = base64.b64encode(image_file.read()).decode("utf-8")

    few_shot_examples_data_3 = [
        {
            "image": encoded_image_3,
            "text_description": "The screen shot shows a the initial screen of the phone. There is an 'Gmail',\
                'Photos' and 'YouTube' apps at the bottom. Below those apps there are \
                'Call', 'Message', 'Chrome' and 'Gmail' apps and below those one the search app bar",
            "action_suggestion": "Plan: Identify the 'Gmail' app. Action: click(id:gmail_button)"
        }
    ]

    generated_prompt_3 = generate_prompt_from_screenshot(
        image_path=example_image_path_3,
        task_id=example_task_id_3, # Pass task_id
        task_goal=example_task_goal_3,
        description_prompt=specific_description_prompt,
        few_shot_examples=few_shot_examples_data_3
    )
    tqdm.write(f"\n--- Example 3 Prompt Generated ---")
    tqdm.write(generated_prompt_3)
    tqdm.write("--------------------------------\n")
    
    # Clean up dummy images
    # if os.path.exists(example_image_path_1):
    #     os.remove(example_image_path_1)
    # if os.path.exists(example_image_path_2):
    #     os.remove(example_image_path_2)
    # if os.path.exists(example_image_path_3):
    #     os.remove(example_image_path_3)
        
        
