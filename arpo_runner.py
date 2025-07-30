# ARPO (Agentic Replay Policy Optimization) is a reinforcement learning (RL) algorithm designed 
# to train GUI agents by optimizing their interaction policy, including the prompts 
# used by an underlying LLM. ARPO uses reinforcement learning to train multi-turn, 
# vision-language GUI agents. ARPO works by having the agent generate actions 
# based on observations and interaction history and receive policy gradients from sparse, 
# delayed rewards. ARPO training delivers significant gains on in-domain tasks. 
# ARPO significantly boosts in-domain task success rates compared to baseline 
# and GRPO-only training. ARPO establishes a new performance baseline for 
# LLM-based GUI agents trained via reinforcement learning. ARPO is a promising method 
# for training autonomous agents that interact with graphical user interfaces.
# We attempt to evaluate out own policy here, a scenario similar to the one that ARPO uses 
# given that we were not able to run the evaluation in the original framework.

import os
import sys
import random

#import androidworld_env # Assuming AndroidWorld environment library exists
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import json # For loading tasks from a JSON file
from tqdm import tqdm
from datetime import datetime

COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"
COLOR_GREEN = "\033[32m"
COLOR_CYAN = "\033[36m"
COLOR_RED = "\033[31m"
COLOR_RESET = "\033[0m"

# --- Configuration ---
RESULTS_DIR = "results"
ARPO_OPTIMIZER_LOG_DIR = os.path.join(RESULTS_DIR, "arpo_logs")
TASK_FILE = "TASK_FILE.json"

# project_root = os.path.dirname(os.path.abspath(__file__))
# print(project_root)
# sys.path.append(project_root)

from android_world.agents import infer
from android_world.agents import t3a
from android_world.agents import text2grad_prompt
from android_world.agents import arpo_reward
from typing import Type
from android_world.env import env_launcher
from android_world import registry
from android_world.task_evals import task_eval
import logging

    
# --- LLM Agent using LangChain and Hugging Face (from previous code) ---
class HFLLMAgent:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", **kwargs): 
        self.generator = pipeline(
            'text-generation', 
            model=model_name, 
            **kwargs
        )
        self.llm = HuggingFacePipeline(pipeline=self.generator)
        self.prompt_template = PromptTemplate.from_template(
            """You are an Android agent. Your goal is to complete the given task by interacting with the Android app.
            You will be provided with the current screen's UI elements and the task description.
            Based on this information, decide the next action to take.

            Current Screen:
            {observation}

            Task: {task_description}

            Your response should be a single, valid Android action in the format:
            click(id: element_id) or input(id: element_id, text: "your text") or scroll(direction: up/down/left/right)
            """
        )

    def generate_action(self, observation, task_description):
        prompt = self.prompt_template.format(observation=observation, task_description=task_description)
        
        action = self.llm.invoke(prompt) 
        return action.strip(), True, action
    
    def generate_summary(self, prompt):
        summary = self.llm.invoke(prompt)
        return summary.strip(), True, summary

# --- Environment Simulation (from previous code) ---
def simulate_action(current_observation, llm_action, task_description):
    new_observation = current_observation.copy()
    done = False
    success = False

    parsed_action = parse_llm_action(llm_action)

    if parsed_action["action_type"] == "click":
        element_id_to_click = parsed_action.get("element_id")
        if "settings" in element_id_to_click and "settings_screen" not in new_observation.get("current_screen_name", ""):
            new_observation["current_screen_name"] = "settings_screen"
            new_observation["ui_elements"] = [
                {"text": "Brightness", "resource_id": "brightness_setting_id"},
                {"text": "Sound", "resource_id": "sound_setting_id"}
            ]
        if "brightness_setting_id" in element_id_to_click and "change the brightness" in task_description:
            done = True
            success = True
    elif parsed_action["action_type"] == "input":
        text_to_input = parsed_action.get("text")
        element_id_to_input = parsed_action.get("element_id")
        if "search_bar_id" in element_id_to_input:
            new_observation["search_query"] = text_to_input
    
    info = {"success": success}
    
    ### Define reward 
    ## 1 if reached goal 0 otherwise
    reward = 1 if success else 0
    
    return new_observation, done, reward, info

def process_android_observation(observation):
    ui_elements_text = ""
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

# --- Main Runner Script ---
def main():
    
    os.makedirs(ARPO_OPTIMIZER_LOG_DIR, exist_ok=True)
    # Initialize AndroidWorld environment (or a mock if truly standalone)
    #env = androidworld_env.AndroidWorldEnv() # Assuming this handles initial state reset
    
    tqdm.write(f"{COLOR_RED} Initialize Android World environment{COLOR_RESET}")
    env = env_launcher.load_and_setup_env(
        console_port=5554,
        emulator_setup=False,
        adb_path=os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
    )
    # Initialize LLM agent
    # llm_agent = HFLLMAgent(model_name="HuggingFaceH4/zephyr-7b-beta") 
    # tqdm.write(f"{COLOR_RED} Initialize LLM agent Model: meta-llama/Llama-3.1-8B-Instruct {COLOR_RESET}")
    # llm_agent = HFLLMAgent(model_name="meta-llama/Llama-3.1-8B-Instruct")

    # Load tasks from a JSON file or define them directly
    # For a real evaluation, you'd have a more robust task loading mechanism
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
        print(f"\n--- Evaluating Task {i+1}/{len(tasks)} ---")
        
    
        # Reset environment for each task
        #current_observation = env.reset(task["goal"]) # Get initial observation
        tqdm.write(f"{COLOR_RED} Reset Android World emulator to home{COLOR_RESET}")
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
        tqdm.write(f"{COLOR_RED} Initialize task in Android World environment{COLOR_RESET}")
        task.initialize_task(env)
    
        print(f"\n--- Evaluating Task {i+1}/{len(tasks)}: {task.goal} ---")
    
        # 5. Initialize agent
        tqdm.write(f"{COLOR_MAGENTA} Initialize agent in World environment{COLOR_RESET}")
        agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))

        done = False
        steps = 0
        max_steps = 100 
        total_reward = 0
        is_done = False
        step = 0
        
        while not done and steps < max_steps:
            steps += 1
            response = agent.step(task.goal)
            if response.done:
                is_done = True
                break
            
            
            if done:
                print(f"  Task completed: {response.done}")
            elif steps >= max_steps:
                print(f"  Max steps reached for task.")
        
        agent_successful = is_done and task.is_successful(env) == 1
        
        success = 1 if agent_successful else 0
        reward = 1 if agent_successful else 0

        task_results = {"success": success, "steps": steps, "reward": reward}
            
        results.append(task_results)
        print(f"  Results for task '{task.goal}': Success={task_results['success']}, Steps={task_results['steps']}, Reward={task_results['reward']}")
        
        task_log = {
                "task": task.goal,
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "success": task_results["success"],
                "steps": task_results["steps"],
                "reward": task_results["reward"]
            }
        # Log task results
        log_filename = f"arpo_runner_task_{task.goal}_{task_log['timestamp']}.json"
        log_filepath = os.path.join(ARPO_OPTIMIZER_LOG_DIR, log_filename)
        with open(log_filepath, "w") as f:
            json.dump(task_log, f, indent=4)
                
    # Analyze and report overall results
    total_successes = sum(1 for r in results if r["success"])
    average_steps = sum(r["steps"] for r in results) / len(results)
    total_reward = sum(r["reward"] for r in results)

    print(f"\n=== Overall Evaluation Results ===")
    print(f"  Total tasks evaluated: {len(tasks)}")
    print(f"  Total successes: {total_successes}")
    print(f"  Success rate: {total_successes / len(tasks) * 100:.2f}%")
    print(f"  Average steps per successful task: {average_steps:.2f}")
    print(f"  Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()

