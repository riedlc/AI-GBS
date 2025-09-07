# reasoning_capture.py
import os
import json
from datetime import datetime

class ReasoningCapture:
    """Captures LLM reasoning traces for analysis without cluttering simulation output"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.reasoning_file = os.path.join(results_dir, "llm_reasoning.json")
        self.reasoning_data = {
            "capture_rounds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # Rounds to capture
            "reasoning_traces": {}
        }
    
    def capture_reasoning(self, round_num: int, agent_id: int, prompt: str, response: str, extracted_guess: int):
        """Capture reasoning trace if it's in the target rounds"""
        if round_num in self.reasoning_data["capture_rounds"]:
            if round_num not in self.reasoning_data["reasoning_traces"]:
                self.reasoning_data["reasoning_traces"][round_num] = {}
            
            self.reasoning_data["reasoning_traces"][round_num][f"agent_{agent_id}"] = {
                "prompt": prompt,
                "response": response,
                "extracted_guess": extracted_guess,
                "timestamp": datetime.now().isoformat()
            }
    
    def save_reasoning(self):
        """Save captured reasoning traces to file"""
        print(f"Saving reasoning traces to: {self.reasoning_file}")
        print(f"Captured rounds: {list(self.reasoning_data['reasoning_traces'].keys())}")
        with open(self.reasoning_file, 'w') as f:
            json.dump(self.reasoning_data, f, indent=2)
        print(f"LLM reasoning traces captured: {self.reasoning_file}")
    
    def print_captured_reasoning(self):
        """Print captured reasoning traces in a readable format"""
        # Removed console display - only save to file
        pass

# Global reasoning capture instance
_reasoning_capture = None

def init_reasoning_capture(results_dir: str):
    """Initialize reasoning capture for a game"""
    global _reasoning_capture
    _reasoning_capture = ReasoningCapture(results_dir)

def capture_reasoning(round_num: int, agent_id: int, prompt: str, response: str, extracted_guess: int):
    """Capture reasoning trace if capture is initialized"""
    global _reasoning_capture
    if _reasoning_capture:
        _reasoning_capture.capture_reasoning(round_num, agent_id, prompt, response, extracted_guess)
    else:
        print(f"Reasoning capture not initialized! Round {round_num}, Agent {agent_id}")

def save_and_display_reasoning():
    """Save captured reasoning traces to file only"""
    global _reasoning_capture
    if _reasoning_capture:
        _reasoning_capture.save_reasoning()
    else:
        print("No reasoning capture instance to save!")
