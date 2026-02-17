# Emergent coordination in multi-agent language models (ICLR'26)

Official implementation of the paper

> Riedl, C. (2026). **Emergent coordination in multi-agent language models.**
> *International Conference on Learning Representations*


```bibtex
@inproceedings{riedl2026multi,
  title     = {Emergent coordination in multi-agent language models},
  author    = {Riedl, Christoph},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://arxiv.org/abs/2510.05174}
}
```

## 🎯 Research Background

This project investigates how groups of AI agents can collectively solve problems through iterative feedback. The experiment is based on the "collective intelligence" paradigm where multiple agents must coordinate their guesses to reach a target number. In particular, our paper analyzes the group guessing data using information theory (TDMI and information decomposition) to test if multi-agent LLM systems have the capacity for emergent synergy.

### Game Mechanics
- **Sum Mode**: Agents try to make their individual guesses sum to a mystery number
- **Mean Mode**: Agents try to make their average guess equal to a mystery number
- **Feedback**: After each round, agents receive feedback on whether their collective result was too high or too low
- **Convergence**: The game ends when agents successfully reach the target or after maximum rounds

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key


### Installation
```bash
# Clone the repository
git clone https://github.com/riedlc/AI-GBS.git
cd AI-GBS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Single Experiment
```bash
# Run a single experiment (5 agents, temperature=1.9, sum mode)
python experiment.py
```

### Large-Scale Experiments
```bash
# Run full parameter sweep (19 agent counts × 21 temperatures × 10 runs each)
python run_experiment_multi_model.py "gpt-4.1-2025-04-14"
```

## 📊 Experiment Parameters

### Parameter Grid
- **Agent Counts**: 2 through 20 agents (19 values)
- **Temperatures**: 0.0 to 2.0 in 0.1 steps (21 values)
- **Runs per Configuration**: 10 (configurable)
- **Total Experiments**: 3,990 configurations

### Model Support
- OpenAI models (GPT-4, GPT-3.5, etc.)
- Customizable via command line argument

## 📁 Output Structure

### Single Experiment
```
results/experiment_run_YYYYMMDD_HHMMSS/
├── config.json              # Experiment configuration
├── game_data.csv            # Round-by-round agent guesses
├── game_log.txt             # Human-readable game log
├── summary.json             # Final results summary
├── llm_prompts.json         # Captured LLM prompts (selected rounds)
├── round_01.json            # Detailed round data
├── round_02.json
└── ...
```

### Large-Scale Experiments
```
results/massive_experiment_MODEL_TIMESTAMP/
├── experiment_config.json   # Full experiment configuration
├── progress.json            # Batch progress tracking
├── sum_a02_t0.0/            # Configuration folders
│   ├── run_001/             # Individual run folders
│   ├── run_002/
│   └── ...
├── sum_a02_t0.1/
└── ...
```

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Key Parameters (in `run_experiment_multi_model.py`)
```python
agents_list = list(range(2, 21))                    # Agent counts
temp_list = [float(i/10) for i in range(0, 21)]     # Temperature range
runs_per_config = 10                                # Runs per config
max_concurrent = 8                                  # Concurrent experiments
batch_size = 20                                     # Configs per batch
```


### Data Formats
- **CSV**: One column per agent, one row per round
- **JSON**: Detailed experiment metadata and results
- **Logs**: Human-readable progress and error tracking

## ⚡ Performance & Scalability

### Concurrent Architecture
- **Async/await**: Non-blocking API calls
- **Agent Concurrency**: All agents in an experiment call API simultaneously
- **Experiment Concurrency**: Multiple experiments run in parallel
- **Batch Processing**: Configurations grouped for efficient execution

### Expected Performance
- **Single Experiment**: ~30 seconds (5 agents, 15 rounds)
- **Full Parameter Sweep**: ~1-2 days (3,990 experiments)
- **API Calls**: ~3M total calls for full experiment

### Resource Usage
- **CPU**: Minimal (async I/O bound)
- **Memory**: Low (streaming data processing)
- **Network**: High (API calls)
- **Storage**: Moderate (JSON/CSV output)

## 🛠️ Advanced Usage

### Remote Execution
```bash
# On remote server with screen
screen -S experiment
source venv/bin/activate
python run_experiment_multi_model.py "gpt-4.1-2025-04-14"

# Detach: Ctrl+A, D
# Reattach: screen -r experiment
```

### Resume Interrupted Experiments
```python
# In run_experiment_multi_model.py
resume_from_batch = 5  # Resume from batch 5
```

### Custom Model Configuration
```bash
# Use different OpenAI models
python run_experiment_multi_model.py "gpt-4o-mini"
python run_experiment_multi_model.py "gpt-3.5-turbo"
```

## 🔍 Error Handling & Robustness

### Built-in Resilience
- **API Retries**: Exponential backoff with jitter
- **Parsing Fallbacks**: Multiple strategies for number extraction
- **Progress Persistence**: Automatic saving of experiment state
- **Graceful Degradation**: Mock responses on repeated failures

### Monitoring
- **Real-time Progress**: Batch completion tracking
- **Error Logging**: Detailed failure analysis
- **Resource Monitoring**: Disk usage and API rate limits

## 📚 Research Applications

### Collective Intelligence
- Study how group size affects problem-solving
- Analyze temperature's impact on coordination
- Investigate convergence patterns

### AI Agent Behavior
- Compare different model capabilities
- Analyze prompt engineering effects
- Study multi-agent communication

### Experimental Methodology
- Large-scale parameter sweeps
- Automated data collection
- Statistical analysis pipelines


### Code Structure
- `experiment.py`: Core game logic and agent implementation
- `run_experiment_multi_model.py`: Large-scale experiment orchestration
- `llm_run.py`: API client with retry logic
- `results_visualization.py`: Data analysis and plotting
- `prompt_capture.py`: LLM prompt logging system


