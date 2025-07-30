# import androidworld_env 
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import json 
import torch # For torch_dtype
import os # For accessing environment variables
from tqdm import tqdm
from datetime import datetime

# Configuration
TASK_FILE = "androidworld_task.json"
COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"
COLOR_GREEN = "\033[32m"
COLOR_CYAN = "\033[36m"
COLOR_RED = "\033[31m"
COLOR_RESET = "\033[0m"

# --- Configuration ---
RESULTS_DIR = "results"
SIMULATED_ARPO_LOG_DIR = os.path.join(RESULTS_DIR, "simulated_arpo_logs")

# --- LLM Agent using LangChain and Hugging Face ---
class HFLLMAgent:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", **kwargs): 
        """
        Initializes the LLM agent using a Hugging Face model through LangChain.
        Requires Hugging Face token to be set as HF_TOKEN environment variable.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            **kwargs: Additional arguments for HuggingFacePipeline.
        """
        # Ensure Hugging Face token is available
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set. Please set your Hugging Face token with access to Llama models.")

        # Create a text generation pipeline using Hugging Face Transformers
        # Use torch.bfloat16 for better performance with modern GPUs
        # device_map="auto" lets Hugging Face manage model placement across devices
        self.generator = pipeline(
            'text-generation', 
            token=hf_token, # Pass the Hugging Face token
            **kwargs
        )
        # Wrap the Hugging Face pipeline in a LangChain LLM
        self.llm = HuggingFacePipeline(pipeline=self.generator)

        # Define a prompt template for the agent's actions
        self.prompt_template = PromptTemplate.from_template(
            """You are an Android agent. Your goal is to complete the given task by interacting with the Android app. You will be provided with the current screen's UI elements and the task description. Based on this information, decide the next action to take.

            Current Screen:
            {observation}

            Task: {task_description}

            Your response should be a single, valid Android action in the format: click(id: element_id) or input(id: element_id, text: "your text") or scroll(direction: up/down/left/right)
            """
        )

    def generate_action(self, observation, task_description):
        prompt = self.prompt_template.format(observation=observation, task_description=task_description)
        action = self.llm.invoke(prompt) 
        return action.strip()

# --- Environment Simulation (from previous code) ---
def simulate_action(current_observation, llm_action, task_description):
    new_observation = current_observation.copy()
    done = False
    success = False

    parsed_action = parse_llm_action(llm_action)
    action_type = parsed_action["action_type"]

    if action_type == "click":
        element_id = parsed_action.get("element_id")
        
        # Scenario: Clicking the 'Settings' button on the home screen
        if element_id == "settings_button_id" and new_observation["current_screen_name"] == "home_screen":
            new_observation["current_screen_name"] = "settings_screen"
            new_observation["ui_elements"] = [
                {"text": "Brightness", "resource_id": "brightness_setting_id", "class_name": "TextView", "bounds": (100, 200, 800, 300)},
                {"text": "Sound", "resource_id": "sound_setting_id", "class_name": "TextView", "bounds": (100, 350, 800, 450)},
                {"text": "Back", "resource_id": "back_button_id", "class_name": "ImageButton"}
            ]
            new_observation["current_context_info"] = "User is now in the Settings menu."
            print("SIMULATION: Navigated to Settings screen.")

        # Scenario: Clicking 'Brightness' in settings to change it (completing a task)
        elif element_id == "brightness_setting_id" and new_observation["current_screen_name"] == "settings_screen":
            if "change the brightness" in task_description:
                done = True
                success = True
                print("SIMULATION: Brightness changed. Task completed.")
            else:
                new_observation["current_screen_name"] = "brightness_detail_screen"
                new_observation["ui_elements"] = [
                    {"text": "Adjust Brightness Slider", "resource_id": "brightness_slider_id", "class_name": "SeekBar"},
                    {"text": "OK", "resource_id": "ok_button_id"}
                ]
                print("SIMULATION: Entered Brightness adjustment screen.")
        
        # Scenario: Clicking back button
        elif element_id == "back_button_id" and new_observation["current_screen_name"] == "settings_screen":
            new_observation["current_screen_name"] = "home_screen"
            new_observation["ui_elements"] = [
                {"text": "Search apps", "resource_id": "search_bar_id"},
                {"text": "Settings", "resource_id": "settings_button_id"},
                {"text": "Gallery", "resource_id": "gallery_icon_id"}
            ]
            print("SIMULATION: Returned to Home screen.")

        else:
            print(f"SIMULATION: Clicked element ID '{element_id}' on screen '{new_observation['current_screen_name']}' - no specific simulation logic defined.")

    elif action_type == "input":
        element_id = parsed_action.get("element_id")
        text_to_input = parsed_action.get("text")

        # Scenario: Inputting into the search bar
        if element_id == "search_bar_id" and new_observation["current_screen_name"] == "home_screen":
            new_observation["search_query"] = text_to_input
            new_observation["ui_elements"].append({"text": f"Search results for: {text_to_input}", "resource_id": "search_results_display"})
            new_observation["current_context_info"] = f"User searched for '{text_to_input}'. Showing results."
            if "search for 'weather'" in task_description.lower() and "weather" in text_to_input.lower():
                done = True # Assuming viewing results is the completion for this task
                success = True
                print(f"SIMULATION: Input '{text_to_input}' into search bar. Task completed.")
            else:
                print(f"SIMULATION: Input '{text_to_input}' into search bar.")

    elif action_type == "scroll":
        direction = parsed_action.get("direction")
        # Simulate scrolling - this is more complex as it depends on what's visible
        # For a basic simulation, you might just update context or add/remove some UI elements.
        print(f"SIMULATION: Scrolled {direction}. UI elements might change but not modeled dynamically here.")
        # Example: if scrolling down reveals more items
        # if direction == "down" and "Load More" not in [e["text"] for e in new_observation["ui_elements"]]:
        #     new_observation["ui_elements"].append({"text": "Load More", "resource_id": "load_more_button"})

    elif action_type == "no_op":
        print(f"SIMULATION: No operation performed or action unparseable: {parsed_action.get('reason', 'N/A')}")
        # Potentially penalize or mark as invalid action if the LLM generates unparseable actions

    info = {"success": success}
    return new_observation, done, info


def process_android_observation(observation):
    ui_elements_text = ""
    print(observation["ui_elements"])
    for element in observation["ui_elements"]:
        ui_elements_text += f"{element['text']} (id: {element['resource_id']}), "
    return f"Current screen: {ui_elements_text}"

def parse_llm_action(action_text):
    if "click(" in action_text:
        element_id = action_text.split("click(")[1].split("id:")[1].strip(" )")
        return {"action_type": "click", "element_id": element_id}
    elif "input(" in action_text:
        parts = action_text.split("input(")[1].strip(" )").split(", text: ")
        element_id = parts[0].split("id:")[1].strip()
        text = parts[1].strip('"')
        return {"action_type": "input", "element_id": element_id, "text": text}
    return {"action_type": "no_op"}

def reset(task_goal: str):
    """
    Resets the environment for a new episode.
    Args:
    task_goal (str): The specific task goal for the current episode.
    This is used to set the internal state.
    Returns:
    tuple: (observation, info_dict)
    """
    steps_taken = 0
    goal_reached_internal = False
    task_goal_internal = task_goal
    mock_task_goals = {
    "task_nav_settings_001": {"initial_screen": "home_screen", "target_screen": "settings_menu", "optimal_path": ["click(settings_icon)"]},
    "task_add_item_001": {"initial_screen": "app_list", "target_element": "add_new_item_button", "optimal_path": ["click(add_new_item_button)"]},
    "task_calc_multiply": {"initial_screen": "home_screen", "target_screen": "calc_result_screen", "optimal_path": ["click(calc_icon)", "click(one)", "click(two)", "click(mul)", "click(five)", "click(equals)"]}
    } 
    # Set initial screen based on the task goal
    current_task_info = mock_task_goals.get(task_goal, {"initial_screen": "home_screen", "optimal_path": []})
    current_screen_context = {"current_screen_name": "initial", "ui_elements": [{"resource_id": current_task_info["initial_screen"], "text": current_task_info["initial_screen"].replace('_', ' ').title()}]}
    tqdm.write(f"{COLOR_GREEN} [Env] Resetting for task: '{task_goal_internal}'. Initial state: {current_screen_context}{COLOR_RESET}")
    info = {"task_goal": task_goal}
    
    return current_screen_context, info

# --- Main Runner Script ---
def main():
    os.makedirs(SIMULATED_ARPO_LOG_DIR, exist_ok=True)
    # Initialize AndroidWorld environment (or a mock if truly standalone)
    #env = androidworld_env.AndroidWorldEnv() 

    # Initialize LLM agent with Llama 3.1 8B Instruct
    llm_agent = HFLLMAgent(model_name="meta-llama/Llama-3.1-8B-Instruct") 

    try:
        with open(TASK_FILE, 'r') as f:
            tasks = json.load(f)
    except FileNotFoundError:
        print(f"Task file '{TASK_FILE}' not found. Using dummy tasks.")
        tasks = [
            {"goal": "Navigate to the settings menu and change the brightness", "file": "settings_brightness.json"},
            {"goal": "Open the calculator app and compute 12 * 5", "file": "calculator_multiply.json"}
        ]

    results = []
    for i, task in enumerate(tasks):
        print(f"\n--- Evaluating Task {i+1}/{len(tasks)}: {task['goal']} ---")
        
        current_observation, info = reset(task["goal"]) 

        done = False
        steps = 0
        max_steps = 100 
        
        while not done and steps < max_steps:
            
            processed_observation = process_android_observation(current_observation)
            print(f"  Step {steps+1}: Current observation: {processed_observation}")
            
            action_text = llm_agent.generate_action(processed_observation, task["goal"])
            print(f"  Agent generates action: {action_text}")
            
            current_observation, done, info = simulate_action(current_observation, action_text, task["goal"])
            
            steps += 1
            if done:
                print(f"  Task completed: {info.get('success', False)}")
            elif steps >= max_steps:
                print(f"  Max steps reached for task.")
        
        task_results = {"success": info.get("success", False), "steps": steps}
        results.append(task_results)
        print(f"  Results for task '{task['goal']}': Success={task_results['success']}, Steps={task_results['steps']}")

    total_successes = sum(1 for r in results if r["success"])
    average_steps = sum(r["steps"] for r in results) / len(results)

    print(f"\n=== Overall Evaluation Results ===")
    print(f"  Total tasks evaluated: {len(tasks)}")
    print(f"  Total successes: {total_successes}")
    print(f"  Success rate: {total_successes / len(tasks) * 100:.2f}%")
    print(f"  Average steps per successful task: {average_steps:.2f}")
    
    result_log = {
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "total_tasks": len(tasks),
                "total_successes": total_successes,
                "success_rate": float(total_successes / len(tasks) * 100),
                "average_steps": float(average_steps)
            }

    # Log epoch results
    log_filename = f"simulated_arpo_opt_task_{result_log['timestamp']}.json"
    log_filepath = os.path.join(SIMULATED_ARPO_LOG_DIR, log_filename)
    with open(log_filepath, "w") as f:
        json.dump(result_log, f, indent=4)
    tqdm.write(f"{COLOR_BLUE} Epoch log saved to {log_filepath}{COLOR_RESET}")

if __name__ == "__main__":
    main()
