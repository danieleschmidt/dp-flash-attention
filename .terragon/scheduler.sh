#!/bin/bash
# Terragon Autonomous SDLC Scheduler
# Continuous execution based on adaptive intervals

set -e

REPO_PATH="/root/repo"
TERRAGON_DIR="$REPO_PATH/.terragon"
LOG_FILE="$TERRAGON_DIR/scheduler.log"

# Ensure log directory exists
mkdir -p "$TERRAGON_DIR"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SCHEDULER] $1" | tee -a "$LOG_FILE"
}

log "Starting Terragon Autonomous SDLC Scheduler"

# Function to run with error handling
run_component() {
    local component=$1
    local script=$2
    
    log "Executing $component..."
    
    if python3 "$TERRAGON_DIR/$script" >> "$LOG_FILE" 2>&1; then
        log "✅ $component completed successfully"
        return 0
    else
        log "❌ $component failed - check logs"
        return 1
    fi
}

# Main execution loop
main() {
    cd "$REPO_PATH"
    
    # Continuous execution based on schedule
    while true; do
        log "Starting autonomous execution cycle"
        
        # Run full autonomous cycle
        if run_component "Autonomous Executor" "autonomous-executor.py"; then
            log "🎯 Execution cycle completed successfully"
        else
            log "⚠️ Execution cycle had issues - continuing"
        fi
        
        # Adaptive sleep based on activity level
        # In a real system, this would be more sophisticated
        SLEEP_INTERVAL=3600  # 1 hour default
        
        log "Next execution in $SLEEP_INTERVAL seconds"
        sleep $SLEEP_INTERVAL
    done
}

# Handle signals for graceful shutdown
cleanup() {
    log "Received shutdown signal - cleaning up"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start main execution
main