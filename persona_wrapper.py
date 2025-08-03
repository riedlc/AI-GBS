import random
from typing import List

class PersonaWrapper:
    
    def __init__(self, persona_file: str = "personas_gpt41.txt"):
        self.personas = self._load_personas(persona_file)
        self.agent_personas = {}
    
    def _load_personas(self, file_path: str) -> List[str]:
        with open(file_path, 'r') as f:
            personas = [line.strip() for line in f if line.strip()]
        
        print(f"✅ Loaded {len(personas)} personas from {file_path}")
        return personas
    
    def assign_personas(self, num_agents: int):
        shuffled_personas = self.personas.copy()
        random.shuffle(shuffled_personas)
        
        self.agent_personas = {i: shuffled_personas[i] for i in range(num_agents)}
        return self.agent_personas
    
    def enhance_prompt(self, agent_id: int, original_prompt: str) -> str:
        if agent_id in self.agent_personas:
            persona = self.agent_personas[agent_id]
            return f"{persona}\n\n{original_prompt}"
        return original_prompt
    
 