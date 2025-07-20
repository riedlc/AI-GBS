# experiment.py
import random
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict
from llm_run import chat
import asyncio
import re

# Custom exception for parsing failures
class ParsingError(Exception):
    """Raised when LLM response cannot be parsed into a valid number"""
    pass

@dataclass
class Round:
    """Store data for each round"""
    round_num: int
    guesses: Dict[int, int]  # agent_id -> guess (now int)
    average: float  # For mean mode: actual average; for sum mode: the sum value
    rounded_average: int  # For mean mode: rounded average; for sum mode: the sum
    feedback: str
    mystery_number: int
    mode: str = "mean"  # Add mode tracking
    prompts_used: Dict[int, str] = None  # agent_id -> prompt
    api_response_ids: Dict[int, str] = None  # agent_id -> response file ID
    parsing_failures: Dict[int, str] = None  # agent_id -> error message for failed parses
    fallback_responses: Dict[int, str] = None  # agent_id -> error message for API fallbacks

class Agent:
    """Represents an LLM agent participant"""
    
    def __init__(self, agent_id: int, model: str, temperature: float):
        self.agent_id = agent_id
        self.model = model
        self.temperature = temperature
        self.guess_history = []
        self.last_successful_guess = None  # Track last successful guess for reuse
    
    async def make_guess(self, round_num: int, game_history: List[Round], guess_range: tuple, mode: str) -> tuple[int, str, object, bool]:
        """Generate a guess based on game history and feedback. Returns (guess, prompt_used, response, parsing_failed)"""
        
        if round_num == 1:
            # First round: random guess
            prompt = f"Generate a random integer between {guess_range[0]} and {guess_range[1]}. Respond with only the number."
        else:
            # Subsequent rounds: use feedback and history
            prompt = self._build_strategic_prompt(game_history, guess_range, mode)
        
        # Get response from LLM (now async)
        is_reasoning_model = ("deepseek" in self.model.lower() or "qwen" in self.model.lower())
        
        response = await chat(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=None if is_reasoning_model else 2
        )
        
        # Extract number from response - handle parsing failures gracefully
        try:
            guess = self._extract_number(response, guess_range)
            self.guess_history.append(guess)
            self.last_successful_guess = guess  # Store successful guess
            return guess, prompt, response, False  # parsing_failed = False
        except ParsingError as e:
            # Reuse previous guess if available
            if self.last_successful_guess is not None:
                reused_guess = self.last_successful_guess
                self.guess_history.append(reused_guess)
                return reused_guess, prompt, response, True  # parsing_failed = True
            else:
                # First round parsing failure - use middle of range as fallback
                fallback_guess = (guess_range[0] + guess_range[1]) // 2
                self.guess_history.append(fallback_guess)
                self.last_successful_guess = fallback_guess
                return fallback_guess, prompt, response, True  # parsing_failed = True
    
    def _build_strategic_prompt(self, game_history: List[Round], guess_range: tuple, mode: str) -> str:
        """Build prompt with game history for strategic guessing"""
        
        game_desc = "sum guessing game" if mode == "sum" else "number guessing game"
        target_desc = "sum to the mystery number" if mode == "sum" else "average to the mystery number"
        
        prompt = f"""You are playing a {game_desc}. Your goal is to help your group {target_desc}.
    Your guess range is {guess_range[0]} to {guess_range[1]}.

    Game History:
    """
        
        for round_data in game_history:
            prompt += f"Round {round_data.round_num}: "
            prompt += f"Your guess: {round_data.guesses[self.agent_id]}\n"
            prompt += f"Result: {round_data.feedback}\n\n"
        
        prompt += f"""Based on this feedback, what should your next guess be?

    Respond with only an integer between {guess_range[0]} and {guess_range[1]}."""
        
        return prompt
    
    def _extract_number(self, response, guess_range: tuple) -> int:
        """Extract number from LLM response - raises ParsingError if parsing fails"""
        return self._extract_number_robust(response, guess_range)
    
    def _extract_number_robust(self, response, guess_range: tuple) -> int:
        """Try multiple parsing strategies for robust number extraction"""
        strategies = [
            self._extract_last_number,
            self._extract_first_number, 
            self._extract_any_number,
            lambda r, g: self._generate_fallback(g)
        ]
        
        for strategy in strategies:
            try:
                return strategy(response, guess_range)
            except ParsingError:
                continue
        
        # If all strategies fail, use fallback
        return self._generate_fallback(guess_range)
    
    def _extract_last_number(self, response, guess_range: tuple) -> int:
        """Extract the last number from response"""
        content = self._get_response_content(response)
        numbers = re.findall(r'\d+', content.strip())
        if numbers:
            guess = int(numbers[-1])
            if guess_range[0] <= guess <= guess_range[1]:
                return guess
        raise ParsingError(f"Agent {self.agent_id}: No valid number found in response")
    
    def _extract_first_number(self, response, guess_range: tuple) -> int:
        """Extract the first number from response"""
        content = self._get_response_content(response)
        numbers = re.findall(r'\d+', content.strip())
        if numbers:
            guess = int(numbers[0])
            if guess_range[0] <= guess <= guess_range[1]:
                return guess
        raise ParsingError(f"Agent {self.agent_id}: No valid number found in response")
    
    def _extract_any_number(self, response, guess_range: tuple) -> int:
        """Extract any number from response, even if outside range"""
        content = self._get_response_content(response)
        numbers = re.findall(r'\d+', content.strip())
        if numbers:
            guess = int(numbers[0])
            # Clamp to valid range
            guess = max(guess_range[0], min(guess, guess_range[1]))
            return guess
        raise ParsingError(f"Agent {self.agent_id}: No number found in response")
    
    def _generate_fallback(self, guess_range: tuple) -> int:
        """Generate a fallback guess"""
        return (guess_range[0] + guess_range[1]) // 2
    
    def _get_response_content(self, response) -> str:
        """Extract content from different response formats"""
        if hasattr(response, 'message'):
            # Ollama format
            return response.message.content
        elif hasattr(response, 'choices'):
            # OpenAI/OpenRouter format
            return response.choices[0].message.content
        else:
            return str(response)

class GameMaster:
    """Orchestrates the number guessing game"""
    
    def __init__(self, mode: str = "mean", mystery_range: tuple = None, temperature: float = 0.7, 
             max_rounds: int = 20, num_agents: int = None, batch_folder: str = None, run_id: int = 1):  
        self.mode = mode  # "sum" or "mean"
        
        # Set default ranges based on mode
        if mystery_range is None:
            if mode == "sum":
                mystery_range = (51, 100)  # Original experiment range
            else:  # mean mode
                mystery_range = (0, 100)
        
        self.mystery_number = random.randint(*mystery_range)
        self.mystery_range = mystery_range
        # Set guess range based on mode
        self.guess_range = (0, 50) if mode == "sum" else mystery_range
        self.temperature = temperature
        self.agents: List[Agent] = []
        self.game_history: List[Round] = []
        self.max_rounds = max_rounds
        
        # Create results directory
        if batch_folder:
            # Batch mode - config folder + run folder
            config_name = f"{mode}_a{num_agents:02d}_t{temperature:.1f}"
            run_folder = f"run_{run_id:03d}"
            self.results_dir = f"{batch_folder}/{config_name}/{run_folder}"
        else:
            # Single mode
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"results/experiment_run_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save run configuration
        self._save_config()
    
    def add_agent(self, model: str) -> Agent:
        """Add an agent to the game"""
        agent_id = len(self.agents)
        agent = Agent(agent_id, model, self.temperature)
        self.agents.append(agent)
        return agent
    
    def _save_config(self):
        """Save run configuration"""
        config = {
            "mystery_number": self.mystery_number,
            "mystery_range": self.mystery_range,
            "guess_range": self.guess_range,
            "mode": self.mode,
            "temperature": self.temperature,
            "max_rounds": self.max_rounds,
            "timestamp": datetime.now().isoformat(),
            "agents": [
                {
                    "agent_id": i,
                    "model": agent.model,
                    "client_type": "openai"
                }
                for i, agent in enumerate(self.agents)
            ]
        }
        
        with open(os.path.join(self.results_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
    
    def _save_round(self, round_data: Round):
        """Save individual round data"""
        filename = os.path.join(self.results_dir, f"round_{round_data.round_num:02d}.json")
        with open(filename, 'w') as f:
            json.dump(asdict(round_data), f, indent=2)
    
    def _update_log(self, round_data: Round):
        """Update simple log file"""
        log_file = os.path.join(self.results_dir, "game_log.txt")
        guesses_vector = list(round_data.guesses.values())
        result_label = "sum" if self.mode == "sum" else "avg"
        
        # Build log line with parsing failure indicators
        log_line = f"Round {round_data.round_num}: guesses={guesses_vector}, {result_label}={round_data.average:.1f}, rounded={round_data.rounded_average}, {round_data.feedback}"
        
        # Add parsing failure info if any
        if round_data.parsing_failures:
            failure_info = ", ".join([f"Agent{aid}:REUSED" for aid in round_data.parsing_failures.keys()])
            log_line += f", parsing_failures=[{failure_info}]"
        
        # Add fallback response info if any
        if round_data.fallback_responses:
            fallback_info = ", ".join([f"Agent{aid}:FALLBACK" for aid in round_data.fallback_responses.keys()])
            log_line += f", api_failures=[{fallback_info}]"
        
        with open(log_file, 'a') as f:
            f.write(log_line + "\n")
    
    def _log_parsing_failure(self, round_num: int, agent_id: int, error_msg: str):
        """Log parsing failure to game log"""
        log_file = os.path.join(self.results_dir, "game_log.txt")
        with open(log_file, 'a') as f:
            f.write(f"Round {round_num}: PARSING FAILURE - Agent {agent_id}: {error_msg}\n")
    
    def _save_parsing_failure(self, round_num: int, agent_id: int, error_msg: str):
        """Save detailed parsing failure information"""
        failure_data = {
            "round_num": round_num,
            "agent_id": agent_id,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "game_config": {
                "mode": self.mode,
                "temperature": self.temperature,
                "num_agents": len(self.agents),
                "mystery_number": self.mystery_number
            }
        }
        
        failure_file = os.path.join(self.results_dir, "parsing_failure.json")
        with open(failure_file, 'w') as f:
            json.dump(failure_data, f, indent=2)
        
    async def play_round(self, round_num: int) -> Round:
        """Play a single round"""
        print(f"\n--- Round {round_num} ---")
        print(f"Mystery number: {self.mystery_number} (hidden from agents)")
        
        # Collect guesses CONCURRENTLY from all agents
        tasks = [
            agent.make_guess(round_num, self.game_history, self.guess_range, self.mode) 
            for agent in self.agents
        ]
        
        # Wait for all agents to respond (no try/catch here anymore)
        results = await asyncio.gather(*tasks)
        
        # Process results
        guesses = {}
        prompts_used = {}
        api_response_ids = {}
        parsing_failures = {}  # Track parsing failures
        fallback_responses = {}  # Track API fallback responses
        
        for i, (guess, prompt, response, parsing_failed) in enumerate(results):
            agent = self.agents[i]
            guesses[agent.agent_id] = guess
            prompts_used[agent.agent_id] = prompt
            
            # Track parsing failure and fallback responses
            if parsing_failed:
                parsing_failures[agent.agent_id] = f"Parsing failed, reused previous guess: {guess}"
                print(f"Agent {agent.agent_id}: {guess} (⚠️ reused due to parsing failure)")
            else:
                print(f"Agent {agent.agent_id}: {guess}")
            
            # Track API fallback responses
            if hasattr(response, 'is_fallback') and response.is_fallback:
                fallback_responses[agent.agent_id] = f"API failed, using fallback guess: {guess}"
                print(f"Agent {agent.agent_id}: {guess} (🚨 API FAILED - using fallback)")
            
            # Save API response
                try:
                    response_id = f"api_r{round_num:02d}_a{agent.agent_id}"
                    api_file = os.path.join(self.results_dir, f"raw_api_{response_id}.json")
                    
                    # Convert response to JSON-serializable format
                    if hasattr(response, 'model_dump'):
                        response_data = response.model_dump()
                    elif hasattr(response, 'to_dict'):
                        response_data = response.to_dict()
                    else:
                        # Manual conversion for OpenAI-style responses
                        response_data = {
                            "id": getattr(response, 'id', None),
                            "object": getattr(response, 'object', None),
                            "created": getattr(response, 'created', None),
                            "model": getattr(response, 'model', None),
                            "choices": [
                                {
                                    "index": choice.index if hasattr(choice, 'index') else i,
                                    "message": {
                                        "role": choice.message.role if hasattr(choice.message, 'role') else None,
                                        "content": choice.message.content if hasattr(choice.message, 'content') else None
                                    },
                                    "finish_reason": getattr(choice, 'finish_reason', None)
                                }
                                for i, choice in enumerate(getattr(response, 'choices', []))
                            ],
                            "usage": getattr(response, 'usage', None).__dict__ if hasattr(getattr(response, 'usage', None), '__dict__') else None
                        }
                    
                    with open(api_file, 'w') as f:
                        json.dump(response_data, f, indent=2)
                    
                    api_response_ids[agent.agent_id] = response_id
                    
                except Exception as e:
                    print(f"Warning: Could not save API response for agent {agent.agent_id}: {e}")
        
        # Calculate result based on mode
        if self.mode == "sum":
            total_sum = sum(guesses.values())
            result_value = total_sum
            average_for_storage = total_sum
            print(f"Group sum: {total_sum}")
        else:  # mean mode
            average = sum(guesses.values()) / len(guesses)
            result_value = round(average)
            average_for_storage = average
            print(f"Group average: {average:.1f} -> rounded: {result_value}")
        
        # Generate feedback
        difference = result_value - self.mystery_number
        if difference == 0:
            feedback = "CORRECT! 🎯"
        elif difference > 0:
            feedback = f"too HIGH by {difference}"
        else:
            feedback = f"too LOW by {abs(difference)}"
        
        print(f"Feedback: {feedback}")
        
        # Store round data
        round_data = Round(
            round_num=round_num, 
            guesses=guesses, 
            average=average_for_storage,
            rounded_average=result_value,
            feedback=feedback, 
            mystery_number=self.mystery_number,
            mode=self.mode,
            prompts_used=prompts_used,
            api_response_ids=api_response_ids if api_response_ids else None,
            parsing_failures=parsing_failures if parsing_failures else None
        )
        
        # Add fallback responses to round data if any
        if fallback_responses:
            round_data.fallback_responses = fallback_responses
        self.game_history.append(round_data)
        
        # Save immediately
        self._save_round(round_data)
        self._update_log(round_data)
        
        return round_data
    
    async def play_game(self) -> List[Round]:
        """Play the complete game until solved or max rounds reached"""
        print(f"🎮 Starting {self.mode.title()} Guessing Game!")
        print(f"Results will be saved to: {self.results_dir}")
        print(f"Mystery number range: {self.mystery_range}")
        print(f"Guess range: {self.guess_range}")
        print(f"Number of agents: {len(self.agents)}")
        print(f"Temperature: {self.temperature}")
        
        # Update config with final agent count
        self._save_config()
        
        round_num = 1
        while round_num <= self.max_rounds:
            round_data = await self.play_round(round_num)
            
            # Check if solved
            if "CORRECT" in round_data.feedback:
                print(f"\n🏆 Game solved in {round_num} rounds!")
                break
            
            round_num += 1
        
        if round_num > self.max_rounds:
            print(f"\n⏰ Game ended after {self.max_rounds} rounds (max limit reached)")
        
        # Save final summary
        self._save_final_summary()
        
        return self.game_history
    
    def _save_final_summary(self):
        """Save final game summary"""
        summary = {
            "total_rounds": len(self.game_history),
            "solved": any("CORRECT" in r.feedback for r in self.game_history) if self.game_history else False,
            "mystery_number": self.mystery_number,
            "mode": self.mode,
            "final_average": self.game_history[-1].average if self.game_history else None,
            "final_rounded_average": self.game_history[-1].rounded_average if self.game_history else None,
            "completed_successfully": True
        }
        
        with open(os.path.join(self.results_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

# Async runner function
async def run_async_test(num_agents: int = 5, model: str = "gpt-4o-mini", 
                   temperature: float = 0.7, mode: str = "mean", run_id: int = 1):
    import time
    start_time = time.time()
    
    game = GameMaster(mode=mode, temperature=temperature, num_agents=num_agents, run_id=run_id)
   
    # Add agents
    for i in range(num_agents):
        game.add_agent(model)
    
    # Play and results are auto-saved
    await game.play_game()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n📁 Results saved in: {game.results_dir}")
    print(f"⏱️  Total execution time: {elapsed_time:.2f} seconds")
    
    return game.results_dir

# Update the __main__ section
if __name__ == "__main__":
    import asyncio
    import time
    
    async def main():
        # Original sum experiment (like Roberts & Goldstone 2011)
        print("Running SUM experiment...")
        await run_async_test(
            num_agents=5,
            model="gpt-4o-mini", 
            temperature=1.9,
            mode="sum"
        )
    
    total_start_time = time.time()
    asyncio.run(main())
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    print(f"\n🏁 Total program execution time: {total_elapsed:.2f} seconds")