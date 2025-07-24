#!/bin/bash

# List of models to test
# "openai/gpt-4o-mini" (already tried)
# "meta-llama/llama-3.3-70b-instruct"
# "google/gemini-2.5-flash-preview-05-20"
# "deepseek/deepseek-chat-v3-0324"
#  "deepseek/deepseek-r1-0528" 

models=(
   "gpt-4.1-2025-04-14"
)

# Create tmux sessions for each model
for model in "${models[@]}"; do
    # Create a safe session name (replace special chars)
    session_name=$(echo "$model" | sed 's/[^a-zA-Z0-9]/_/g')
    
    echo "🚀 Starting tmux session: $session_name for model: $model"
    
    # Create new tmux session and run the experiment
    tmux new-session -d -s "$session_name" \
        "python run_experiment_multi_model.py '$model'; echo 'Experiment completed. Press any key to exit.'; read"
done

echo ""
echo "✅ All tmux sessions started!"
echo ""
echo "📋 Session list:"
tmux list-sessions

echo ""
echo "🔗 To attach to a session, use:"
for model in "${models[@]}"; do
    session_name=$(echo "$model" | sed 's/[^a-zA-Z0-9]/_/g')
    echo "   tmux attach-session -t $session_name"
done

echo ""
echo "🛑 To kill all sessions when done:"
echo "   tmux kill-server"