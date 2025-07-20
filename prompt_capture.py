# prompt_capture.py
import os
import json
from datetime import datetime

class PromptCapture:
    """Captures LLM prompts for analysis without cluttering simulation output"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.prompts_file = os.path.join(results_dir, "llm_prompts.json")
        self.prompts_data = {
            "capture_rounds": [1, 2, 5, 10, 15, 20],  # Rounds to capture
            "prompts": {}
        }
    
    def capture_prompt(self, round_num: int, agent_id: int, prompt: str):
        """Capture a prompt if it's in the target rounds"""
        if round_num in self.prompts_data["capture_rounds"]:
            if round_num not in self.prompts_data["prompts"]:
                self.prompts_data["prompts"][round_num] = {}
            
            self.prompts_data["prompts"][round_num][f"agent_{agent_id}"] = {
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            }
    
    def save_prompts(self):
        """Save captured prompts to file"""
        with open(self.prompts_file, 'w') as f:
            json.dump(self.prompts_data, f, indent=2)
        print(f"📝 LLM prompts captured: {self.prompts_file}")
    
    def print_captured_prompts(self):
        """Print captured prompts in a readable format"""
        # Removed console display - only save to file
        pass

# Global prompt capture instance
_prompt_capture = None

def init_prompt_capture(results_dir: str):
    """Initialize prompt capture for a game"""
    global _prompt_capture
    _prompt_capture = PromptCapture(results_dir)

def capture_prompt(round_num: int, agent_id: int, prompt: str):
    """Capture a prompt if capture is initialized"""
    global _prompt_capture
    if _prompt_capture:
        _prompt_capture.capture_prompt(round_num, agent_id, prompt)

def save_and_display_prompts():
    """Save captured prompts to file only"""
    global _prompt_capture
    if _prompt_capture:
        _prompt_capture.save_prompts() 