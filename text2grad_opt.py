# text2grad_opt.py

import os
import json
import sys
from datetime import datetime
import random # For simulating prompt perturbations
from tqdm import tqdm
import argparse

from android_world.agents import infer
from android_world.agents import t3a
from android_world.agents import text2grad_prompt
from typing import Type
from android_world.env import env_launcher
from android_world import registry
from android_world.task_evals import task_eval
import logging

# project_root = os.path.dirname(os.path.abspath(__file__))
# print(project_root)
# sys.path.append(project_root)

COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"
COLOR_GREEN = "\033[32m"
COLOR_CYAN = "\033[36m"
COLOR_RED = "\033[31m"
COLOR_RESET = "\033[0m"

os.environ['GRPC_TRACE'] = 'none'  # Disable tracing
os.environ['GRPC_VERBOSITY'] = 'NONE'

logging.getLogger('grpc').setLevel(logging.ERROR)

# --- Configuration ---
RESULTS_DIR = "results"
TEXT2GRAD_OPTIMIZER_LOG_DIR = os.path.join(RESULTS_DIR, "text2grad_logs")
SIMULATION = 'INIT_RANDOM_SCORE'


def run_evaluation(llm_agent_instance, simulated_env_instance, task, task_goal, prompt_override) -> (dict):
    # 1. Initialize environment
    tqdm.write(f"{COLOR_MAGENTA} Initialize Android World environment{COLOR_RESET}")
    env = env_launcher.load_and_setup_env(
        console_port=5554,
        emulator_setup=False,
        adb_path=os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
    )
    # 2. Reset emulator to home
    tqdm.write(f"{COLOR_MAGENTA} Reset Android World emulator to home{COLOR_RESET}")
    env.reset(go_home=True)
    # 3. Manipulate tasks
    tqdm.write(f"{COLOR_MAGENTA} Manipulate tasks in Android World environment{COLOR_RESET}")
    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
    if task:
        if task not in aw_registry:
            raise ValueError('Task {} not found in registry.'.format(task))
        task_type: Type[task_eval.TaskEval] = aw_registry[task]
    else:
        task_type: Type[task_eval.TaskEval] = random.choice(
            list(aw_registry.values())
        )
    params = task_type.generate_random_params()
    task = task_type(params)
    # 4. Initialize task
    tqdm.write(f"{COLOR_MAGENTA} Initialize task in Android World environment{COLOR_RESET}")
    task.initialize_task(env)
    # 5. Initialize agent
    tqdm.write(f"{COLOR_MAGENTA} Initialize Text2gradPrompt agent in World environment{COLOR_RESET}")
    agent = text2grad_prompt.Text2gradPrompt(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
    
    print('Goal: ' + str(task.goal))
    is_done = False
    step = 0
    for _ in range(int(task.complexity * 10)):
        step += 1
        print(step, prompt_override)
        print(int(task.complexity * 10))
        tqdm.write(f"{COLOR_MAGENTA} For loop steps: {int(task.complexity * 10):.2f}{COLOR_RESET}")
        # 6. Call the step function with a new prompt <prompt_override>
        response = agent.step(task.goal, prompt_override)
        if response.done:
            is_done = True
            break
        
        
    agent_successful = is_done and task.is_successful(env) == 1
    print(f'{"Task Successful ✅" if agent_successful else "Task Failed ❌"};'f' {task.goal}')
    # 7. Close the environment
    env.close()
    success = 1 if agent_successful else 0
    results = {"success": success, "steps": step}
    print(results)
    return results

# --- Placeholder for the evaluation function (would be in main_evaluation_script.py or utils) ---
# This function simulates running the LLM agent with a given prompt in the environment.
# In a real scenario, this would execute your LLM agent against the AndroidEnv.
def evaluate_prompt_performance(prompt: str, task_id: str, task:str, task_goal:str, llm_agent_instance, simulated_env_instance) -> float:
    """
    Simulates evaluating a prompt by running the LLM agent with it on a task.
    Returns a scalar performance metric (e.g., success rate or reward).
    """
    tqdm.write(f"{COLOR_MAGENTA}  [Text2Grad] Evaluating prompt for Task '{task_id}'...{COLOR_RESET}")
    # This is a placeholder. You would call your actual evaluation loop here.
    # For example:
    if SIMULATION == 'ANDROID_WORLD':
        tqdm.write(f"{COLOR_MAGENTA} Before run_evaluation Android World environment{COLOR_RESET}")
        results = run_evaluation(llm_agent_instance, simulated_env_instance, task, task_goal=task_goal, prompt_override=prompt)
        
        performance_score = float(results['success'] * 10 - results['steps'] * 0.1) # Example scoring
    
    # Simulate performance - better prompts get higher scores
    # This is where the LLM actually tries to solve the task with the generated prompt
    # and we get a quantifiable performance score.
    # For demonstration, we'll assign a random score.
    if SIMULATION == "INIT_RANDOM_SCORE":
        performance_score = random.uniform(0.1, 10.0) # Placeholder: actual score from environment interaction
    tqdm.write(f"{COLOR_MAGENTA}  [Text2Grad] Evaluation completed. Score: {performance_score:.2f}{COLOR_RESET}")
    return performance_score


# --- Text2Grad Optimizer Simulation ---
class Text2GradOptimizer:
    """
    A placeholder class to simulate Text2Grad-like prompt optimization.
    In a real scenario, this would interface with the actual Text2Grad library or API.
    """
    def __init__(self, llm_agent_instance, simulated_env_instance, **kwargs):
        self.llm_agent = llm_agent_instance
        self.simulated_env = simulated_env_instance
        self.perturbation_strength = kwargs.get("perturbation_strength", 0.1) # How much to change prompts
        self.optimizer_config = kwargs # Store other config

    def _generate_prompt_perturbations(self, base_prompt: str, num_variations: int = 5) -> list[str]:
        """
        Simulates generating variations of a base prompt.
        In a real Text2Grad, this would use the framework's gradient-based suggestions.
        """
        tqdm.write(f"{COLOR_BLUE}    [Text2Grad] Generating {num_variations} prompt perturbations...{COLOR_RESET}")
        perturbed_prompts = []
        for _ in range(num_variations):
            # Very simplistic perturbation: add a random phrase or rephrase slightly
            # A real Text2Grad would use gradients to guide these perturbations
            modifiers = [
                "Ensure clarity.", "Be concise.", "Focus on buttons.",
                "Prioritize scrolling.", "Consider all UI elements."
            ]
            random_modifier = random.choice(modifiers)
            # Simple addition, a real implementation would be more sophisticated
            perturbed_prompt = base_prompt + f" {random_modifier}" if random.random() > 0.5 else base_prompt.replace("optimal action", f"best action considering {random_modifier.lower()}")
            perturbed_prompts.append(perturbed_prompt)
        return perturbed_prompts

    def optimize_prompt(
        self,
        initial_prompt: str,
        task_id: str,
        task:str,
        task_goal:str,
        epochs: int = 5
    ) -> (str, list[dict]):
        """
        Simulates the black-box optimization loop to find an improved prompt.

        Args:
            initial_prompt (str): The prompt to start optimization from (e.g., from Gemini).
            task_id (str): Unique identifier for the task.
            epochs (int): Number of optimization iterations.

        Returns:
            tuple: (best_prompt_found, optimization_history)
        """
        tqdm.write(f"{COLOR_BLUE}[Text2Grad] Starting optimization for Task '{task_id}'...{COLOR_RESET}")
        os.makedirs(TEXT2GRAD_OPTIMIZER_LOG_DIR, exist_ok=True)

        best_prompt = initial_prompt
        best_score = -float('inf')
        optimization_history = []

        current_prompt_version = initial_prompt # The prompt currently being optimized

        for epoch in range(epochs):
            tqdm.write(f"{COLOR_BLUE}  [Text2Grad] Epoch {epoch + 1}/{epochs}{COLOR_RESET}")
            prompt_variations = self._generate_prompt_perturbations(current_prompt_version, num_variations=3)
            
            # Include the current best prompt in the variations to re-evaluate it
            # prompt_variations.append(current_prompt_version) 

            epoch_scores = []
            for i, prompt_var in enumerate(prompt_variations):
                # Evaluate each prompt variation
                score = evaluate_prompt_performance(prompt_var, task_id, task, task_goal, self.llm_agent, self.simulated_env)
                epoch_scores.append({"prompt_version": i, "prompt": prompt_var, "score": score})

                if score > best_score:
                    best_score = score
                    best_prompt = prompt_var
                    tqdm.write(f"{COLOR_YELLOW}    [Text2Grad] New best prompt found (Score: {best_score:.2f}): {best_prompt}{COLOR_RESET}")
            
            # Update the current prompt version for the next epoch based on the best from this epoch
            current_prompt_version = best_prompt 

            epoch_log = {
                "epoch": epoch + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "base_prompt_this_epoch": current_prompt_version,
                "prompt_variations_evaluated": epoch_scores,
                "best_prompt_this_epoch": best_prompt,
                "best_score_this_epoch": best_score
            }
            optimization_history.append(epoch_log)

            # Log epoch results
            log_filename = f"text2grad_opt_task_{task_id}_epoch_{epoch+1}_{epoch_log['timestamp']}.json"
            log_filepath = os.path.join(TEXT2GRAD_OPTIMIZER_LOG_DIR, log_filename)
            with open(log_filepath, "w") as f:
                json.dump(epoch_log, f, indent=4)
            tqdm.write(f"{COLOR_BLUE}    [Text2Grad] Epoch log saved to {log_filepath}{COLOR_RESET}")

        tqdm.write(f"{COLOR_BLUE}[Text2Grad] Optimization completed for Task '{task_id}'. Best score: {best_score:.2f}{COLOR_RESET}")
        return best_prompt, optimization_history

def param_parser():
    """
        rgparse.ArgumentParser: Initializes the argument parser.
        add_argument: Defines the argument (e.g., --task).
        type: Specifies the data type (e.g., str for a string).
        required: Ensures the argument is mandatory.
        help: Provides a description for the argument.
        parse_args: Parses the arguments passed via the command line.
        Accessing Arguments: Use args.<argument_name> to access the value.
        Running the Script:
        Save the script as script.py and run it in the terminal like this:

        python script.py --task ContactsAddContact
    """
    # Create the parser
    parser = argparse.ArgumentParser(description="A script that accepts a command-line argument.")

    # Add an argument
    parser.add_argument('--task', type=str, required=True, help="Task to run in Android World")

    # Parse the arguments
    args = parser.parse_args()

    # Use the argument
    #print(f"Hello, {args.task}!")
    return args.task

if __name__ == "__main__":
    # Have the task parameter from argument parser
    task = param_parser()
    # Ensure results directories exist
    os.makedirs(TEXT2GRAD_OPTIMIZER_LOG_DIR, exist_ok=True)

    # --- Dummy LLM Agent and Simulated Environment Instances ---
    # In a real scenario, you would pass actual instances of your HFLLMAgent
    # and SimulatedAndroidWorldEnv classes here.
    class DummyLLMAgent:
        def __init__(self, name="DummyLLM"):
            self.name = name
        # You might need to add a predict or generate_action method that accepts prompts
        def generate_action(self, prompt, *args, **kwargs):
            return "simulated_action", None, "simulated_response"

    class DummySimulatedEnv:
        def __init__(self, name="DummyEnv"):
            self.name = name
        def reset(self, task_goal):
            return {"ui_elements": []}, {}
        def step(self, action):
            return {"ui_elements": []}, 0, True, {"success": False} # Always fail after one step

    dummy_llm_agent = DummyLLMAgent()
    dummy_simulated_env = DummySimulatedEnv()

    initial_gemini_prompt = """Current UI Description and Suggested Actions:
    \nUI Element Analysis:\n\n* **Search Bar:**  A Google search bar is at the top.
    \n* **Icons (Top Row):**  Microphone icon, Camera icon.
    \n* **App Icons (Grid):** Various app icons are arranged in a grid.  
    These include (but are not limited to) YouTube (red play button), Settings (gear), 
    Photos (multi-colored pinwheel), Clock (clock hands), Calendar (calendar page with \"16\"), 
    Camera (camera lens), Chrome (multi-colored circle), Contacts (silhouette of a person), 
    Drive (triangle), Files (folder), Gmail (envelope with \"M\"), Google (G), Maps (location pin), 
    Messages (speech bubble), Phone (phone receiver), and T-Mobile (SIM card with gear).
    \n* **App Labels:** Below each app icon is the corresponding app name.\n* **\"All apps\" 
    Button/Label:** Below the first row of apps, the text \"All apps\" suggests the existence of more 
    apps not currently visible. It could be a button, but likely indicates more apps are available 
    if the user scrolls down.
    \n* **YT Music Icon:** A red circular icon with a white play button and the label \"YT Music\" is at the bottom.
    \n* **Home Bar:** A horizontal bar at the very bottom indicates the home screen navigation area.
    \n\n\nPlan to Achieve the Goal ('Click the 'Add New Item' button'):\n\nThe provided screenshot 
    does *not* contain an \"Add New Item\" button. Therefore, the task cannot be completed on this screen. 
    \n\nIf the \"Add New Item\" button is expected to appear in the context of a specific app, 
    the plan would be:\n\n1. **Identify the target app:**  Determine the app where the \"Add New Item\" button 
    is expected.\n2. **Locate the app icon:**  Find the corresponding icon on the current screen.
    \n3. **Tap the app icon:**  Launch the target application by tapping its icon. 
    \n4. **Locate 'Add New Item' button:**  Once the app loads, identify the  \"Add New Item\" button.
    \n5. **Tap 'Add New Item' button:** Execute the task by tapping the button.\n\n\nIf the \"Add New Item\" 
    button is related to adding an app or widget to the home screen:
    \n\n1. **Long press on the home screen:** Long pressing on an empty space of the home screen usually 
    brings up options to add widgets, apps, and change wallpaper.\n2. **Locate 'Add New Item' button 
    (or equivalent options):** Look for buttons or options that allow adding apps or widgets. 
    These options may vary based on the Android launcher being used.\n3. **Tap 'Add New Item' 
    button (or select the app/widget):** Tap the \"Add New Item\" button if present, or directly select 
    the app or widget to be added.\n\n\nWithout more context about where the \"Add New Item\" button should be, 
    it's impossible to provide more precise instructions.\n\n\nTask Goal: Click the 'Add New Item' button.
    \n\nBased on the above, provide the optimal action as before:"""
    example_task_id = "create_new_contact_task"
    task_goal= "Create a new contact for Emma Ahmed. Their number is +12205547842."

    optimizer = Text2GradOptimizer(dummy_llm_agent, dummy_simulated_env, perturbation_strength=0.1)
    
    best_prompt_found, history = optimizer.optimize_prompt(
        initial_gemini_prompt,
        example_task_id,
        task,
        task_goal,
        epochs=3
    )

    tqdm.write(f"\n--- Text2Grad Optimization Summary ---")
    tqdm.write(f"Initial Prompt: {initial_gemini_prompt}")
    tqdm.write(f"Best Prompt Found: {best_prompt_found}")
    tqdm.write(f"Optimization History Length: {len(history)}")
    tqdm.write("-------------------------------------\n")

