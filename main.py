import streamlit as st
import json
import random
import numpy as np
from datetime import datetime, timedelta, timezone
import math
import io # Required for download button string data

# --- Simulation Helper Functions ---

def generate_random_obstacle(env_w, env_h, max_obstacle_size_perc=0.15):
    """Generates a random obstacle within environment bounds."""
    max_w = int(env_w * max_obstacle_size_perc)
    max_h = int(env_h * max_obstacle_size_perc)
    
    # Ensure minimum size
    obs_w = random.randint(max(1, int(max_w * 0.2)), max_w) 
    obs_h = random.randint(max(1, int(max_h * 0.2)), max_h)
    
    # Ensure obstacle fits within bounds
    obs_x = random.randint(0, env_w - obs_w)
    obs_y = random.randint(0, env_h - obs_h)
    
    return (obs_w, obs_h, obs_x, obs_y)

def is_colliding(box1_x, box1_y, box1_w, box1_h, box2_x, box2_y, box2_w, box2_h):
    """Checks if two rectangular boxes collide."""
    return (box1_x < box2_x + box2_w and
            box1_x + box1_w > box2_x and
            box1_y < box2_y + box2_h and
            box1_y + box1_h > box2_y)

def get_next_position(current_x, current_y, current_w, current_h, 
                      target_x, target_y, speed, 
                      env_w, env_h, obstacles):
    """Calculates the next position towards a target, avoiding obstacles and bounds."""
    
    dx = target_x - current_x
    dy = target_y - current_y
    distance = math.sqrt(dx**2 + dy**2)

    if distance < speed: # Reached target or close enough
        return None, None # Signal to pick a new target

    # Proposed move
    move_x = current_x + (dx / distance) * speed
    move_y = current_y + (dy / distance) * speed
    
    # --- Boundary Collision Check ---
    collision_boundary = False
    if move_x < 0:
        move_x = 0
        collision_boundary = True
    elif move_x + current_w > env_w:
        move_x = env_w - current_w
        collision_boundary = True
        
    if move_y < 0:
        move_y = 0
        collision_boundary = True
    elif move_y + current_h > env_h:
        move_y = env_h - current_h
        collision_boundary = True

    # --- Obstacle Collision Check ---
    collision_obstacle = False
    proposed_box = (int(move_x), int(move_y), current_w, current_h)
    for obs_w, obs_h, obs_x, obs_y in obstacles:
        if is_colliding(proposed_box[0], proposed_box[1], proposed_box[2], proposed_box[3], 
                        obs_x, obs_y, obs_w, obs_h):
            collision_obstacle = True
            break # Collision detected with one obstacle is enough

    if collision_boundary or collision_obstacle:
        # Simple collision response: Stop moving towards target for this frame 
        # A more complex response would involve pathfinding or bouncing, 
        # but for basic simulation, stopping or picking a new target works.
        # Let's signal to pick a new target on collision
         return None, None # Pick a new random target

    return int(move_x), int(move_y)


def generate_synthetic_data(config):
    """Generates the synthetic data based on configuration."""
    
    env_w = config['env_dims'][0]
    env_h = config['env_dims'][1]
    num_actors = config['num_actors']
    duration_seconds = config['duration_seconds']
    framerate = config['framerate']
    num_obstacles = config['num_obstacles']
    min_actor_w = config['min_actor_w']
    max_actor_w = config['max_actor_w']
    min_actor_h = config['min_actor_h']
    max_actor_h = config['max_actor_h']
    min_speed = config['min_speed']
    max_speed = config['max_speed']
    
    total_frames = duration_seconds * framerate
    
    # --- Generate Environment ---
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(seconds=duration_seconds)
    
    obstacles = [generate_random_obstacle(env_w, env_h) for _ in range(num_obstacles)]
    
    environment_data = {
        'dims': (env_w, env_h),
        'obs': obstacles,
        'framerate': framerate,
        'sampling': framerate, # Assuming sampling rate = framerate for now
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'chunk_time': config.get('chunk_time', duration_seconds) # Use duration if chunk_time not specified
    }
    
    # --- Initialize Actors ---
    actors = {}
    actor_states = {} # Internal state for simulation

    for i in range(num_actors):
        actor_id = i
        actor_w = random.randint(min_actor_w, max_actor_w)
        actor_h = random.randint(min_actor_h, max_actor_h)
        
        # Ensure actor starts in a valid position (not inside an obstacle)
        while True:
            start_x = random.randint(0, env_w - actor_w)
            start_y = random.randint(0, env_h - actor_h)
            valid_start = True
            for obs_w, obs_h, obs_x, obs_y in obstacles:
                 if is_colliding(start_x, start_y, actor_w, actor_h, obs_x, obs_y, obs_w, obs_h):
                     valid_start = False
                     break
            if valid_start:
                break

        # Random entry frame
        # Ensure actor can exist for at least 1 frame after entry for exit calculation
        max_possible_entry = max(0, total_frames - 2) 
        # Handle case where total_frames itself is very small
        if max_possible_entry < 0: max_possible_entry = 0 # Should not happen with UI limits but safe guard
        
        # Ensure entry frame calculation is valid if max_possible_entry is 0
        if max_possible_entry == 0:
            entry_frame = 0
        else:
            entry_frame = random.randint(0, max_possible_entry)

        # --- Corrected Duration and Exit Frame Calculation ---
        min_duration_frames = framerate * 2 # Minimum frames lifespan (e.g., 2 seconds)
        # Ensure minimum duration is at least 1 frame
        min_duration_frames = max(1, min_duration_frames) 

        # Calculate max possible duration from entry frame until the end of simulation
        max_possible_duration = total_frames - entry_frame 
        
        # Ensure max_possible_duration is at least 1
        max_possible_duration = max(1, max_possible_duration)

        # Determine the actual duration
        if max_possible_duration < min_duration_frames:
            # Cannot meet min duration, stay for the max possible time
            actual_duration_frames = max_possible_duration
        else:
            # Can meet min duration, pick randomly between min requirement and max possible
            actual_duration_frames = random.randint(min_duration_frames, max_possible_duration)

        # Calculate exit frame (inclusive index of the last frame the actor exists)
        # Subtract 1 because duration includes the entry frame itself
        exit_frame = entry_frame + actual_duration_frames - 1
        
        # Clamp exit frame to the last valid frame index of the simulation
        exit_frame = min(exit_frame, total_frames - 1)
        
        # Ensure exit_frame is at least the entry_frame (handles duration=1 case)
        exit_frame = max(entry_frame, exit_frame)
        # --- End of Corrected Section ---


        actors[actor_id] = {
            'descriptor': [f'person_{actor_id}'], # Simple descriptor
            'entry': [entry_frame], # Store as list for potential multiple entries later
            'exit': [exit_frame],   # Store as list for potential multiple exits later (USE CORRECTED exit_frame)
            'frames': [],         # List of (w, h, x, y) tuples per frame
            'postures': [],       # Placeholder for postures (KIV)
        }

        actor_states[actor_id] = {
            'id': actor_id,
            'w': actor_w,
            'h': actor_h,
            'x': start_x,
            'y': start_y,
            'speed': random.uniform(min_speed, max_speed),
            'target_x': random.randint(0, env_w - actor_w),
            'target_y': random.randint(0, env_h - actor_h),
            'entry_frame': entry_frame,
            'exit_frame': exit_frame, # USE CORRECTED exit_frame
            'active': False # Will become active at entry frame
        }

    # --- Simulation Loop ---
    for frame_idx in range(total_frames):
        for actor_id, state in actor_states.items():
            
            # Check if actor should be active
            if not state['active'] and frame_idx >= state['entry_frame']:
                state['active'] = True
                # Find first valid position if entry frame > 0
                if frame_idx > 0: 
                    # Re-validate starting position if simulation started earlier
                    valid_pos = False
                    attempts = 0
                    while not valid_pos and attempts < 100: # Avoid infinite loop
                        temp_x = random.randint(0, env_w - state['w'])
                        temp_y = random.randint(0, env_h - state['h'])
                        is_clear = True
                        for obs_w, obs_h, obs_x, obs_y in obstacles:
                            if is_colliding(temp_x, temp_y, state['w'], state['h'], obs_x, obs_y, obs_w, obs_h):
                                is_clear = False
                                break
                        if is_clear:
                            state['x'] = temp_x
                            state['y'] = temp_y
                            valid_pos = True
                        attempts += 1
                    if not valid_pos: # Fallback if no clear spot found quickly
                         state['x'] = 0 
                         state['y'] = 0 

            if state['active'] and frame_idx >= state['exit_frame']:
                 state['active'] = False
                 # Record the actual exit frame if needed (can differ slightly due to logic)
                 # actors[actor_id]['exit'][-1] = frame_idx # Update last exit frame if needed

            if not state['active']:
                # Actor is not present in this frame, add placeholder or skip
                # To strictly match example, add () - but this might be ambiguous
                # Let's add None and filter later if needed, or just don't add to 'frames' list if inactive.
                # For simplicity and matching the output format request, we'll only add frames when active.
                continue 

            # --- Actor Movement ---
            current_x, current_y = state['x'], state['y']
            target_x, target_y = state['target_x'], state['target_y']
            
            next_x, next_y = get_next_position(current_x, current_y, state['w'], state['h'], 
                                               target_x, target_y, state['speed'],
                                               env_w, env_h, obstacles)

            if next_x is None or next_y is None: # Reached target or collided, pick new target
                # Pick a new random target that is not inside an obstacle
                valid_target = False
                attempts = 0
                while not valid_target and attempts < 50: # Avoid infinite loop
                    new_target_x = random.randint(0, env_w - state['w'])
                    new_target_y = random.randint(0, env_h - state['h'])
                    is_clear = True
                    # Check if target *area* overlaps obstacle (crude check)
                    for obs_w, obs_h, obs_x, obs_y in obstacles:
                        if is_colliding(new_target_x, new_target_y, state['w'], state['h'], obs_x, obs_y, obs_w, obs_h):
                            is_clear = False
                            break
                    if is_clear:
                         state['target_x'] = new_target_x
                         state['target_y'] = new_target_y
                         valid_target = True
                    attempts += 1
                # If failed to find valid target, keep old one (might oscillate)
                
                # Stay in current position for this frame if no valid move/new target found
                next_x, next_y = current_x, current_y 
            
            # Update state
            state['x'] = next_x
            state['y'] = next_y
            
            # Record frame data for the actor
            frame_data = (state['w'], state['h'], state['x'], state['y'])
            actors[actor_id]['frames'].append(frame_data)
            actors[actor_id]['postures'].append(0) # Placeholder posture '0' (e.g., standing/walking)


    # --- Final Data Structure ---
    # Clean up actors with no frames (if entry/exit logic prevented any)
    final_actors = {aid: data for aid, data in actors.items() if data['frames']}

    # Adjust entry/exit frame numbers if needed based on actual frames recorded
    # (This simple version assumes entry/exit are precise, which might need refinement
    # depending on how inactive periods should be represented in 'frames')
    # For now, we trust the initial entry/exit frame numbers.

    output_data = {
        'environment': environment_data,
        'actor': final_actors
    }
    
    return output_data

# --- Streamlit UI ---

st.set_page_config(layout="wide")

st.title("Synthetic Digital Twin Camera Data Generator")
st.markdown("""
Configure the parameters below to generate synthetic motion data for multiple actors 
within a defined environment. The output JSON follows the specified format.
""")

# --- Configuration Sidebar ---
st.sidebar.header("Configuration")

# Environment Settings
st.sidebar.subheader("Environment Settings")
env_width = st.sidebar.number_input("Environment Width", min_value=50, value=800, step=50)
env_height = st.sidebar.number_input("Environment Height", min_value=50, value=600, step=50)
num_obstacles = st.sidebar.slider("Number of Random Obstacles", min_value=0, max_value=20, value=5)
framerate = st.sidebar.slider("Framerate (fps)", min_value=1, max_value=60, value=10, help="Simulation and sampling rate.")
# sampling_rate = st.sidebar.slider("Sampling Rate (fps)", min_value=1, max_value=60, value=30) # Add if needed separately
duration_seconds = st.sidebar.number_input("Simulation Duration (seconds)", min_value=5, max_value=600, value=30, step=5)
# chunk_time = st.sidebar.number_input("Chunk Time (seconds)", min_value=10, value=600) # Less critical for generation

# Actor Settings
st.sidebar.subheader("Actor Settings")
num_actors = st.sidebar.slider("Number of Actors", min_value=1, max_value=50, value=10)
min_actor_w = st.sidebar.slider("Min Actor Width", min_value=5, max_value=50, value=15)
max_actor_w = st.sidebar.slider("Max Actor Width", min_value=5, max_value=80, value=30)
min_actor_h = st.sidebar.slider("Min Actor Height", min_value=5, max_value=50, value=15)
max_actor_h = st.sidebar.slider("Max Actor Height", min_value=5, max_value=80, value=30)
min_speed = st.sidebar.slider("Min Actor Speed (units/frame)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
max_speed = st.sidebar.slider("Max Actor Speed (units/frame)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)

# Ensure min <= max
if min_actor_w > max_actor_w: max_actor_w = min_actor_w
if min_actor_h > max_actor_h: max_actor_h = min_actor_h
if min_speed > max_speed: max_speed = min_speed

# --- Generate Button and Output ---
st.header("Generate Data")

if st.button("Generate Synthetic Data"):
    
    config = {
        'env_dims': (env_width, env_height),
        'num_obstacles': num_obstacles,
        'framerate': framerate,
        'duration_seconds': duration_seconds,
        'num_actors': num_actors,
        'min_actor_w': min_actor_w,
        'max_actor_w': max_actor_w,
        'min_actor_h': min_actor_h,
        'max_actor_h': max_actor_h,
        'min_speed': min_speed,
        'max_speed': max_speed,
        # 'chunk_time': chunk_time # Add if needed
    }

    with st.spinner("Generating simulation data... Please wait."):
        generated_data = generate_synthetic_data(config)

    st.success("Data Generation Complete!")

    # Display JSON output
    st.subheader("Generated JSON Data")
    st.json(generated_data, expanded=False) # Show collapsed by default

    # Provide download button
    st.subheader("Download Data")
    json_string = json.dumps(generated_data, indent=4)
    
    # Create a filename based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"synthetic_camera_data_{timestamp}.json"

    # Use BytesIO for download button compatibility with text
    json_bytes = io.BytesIO(json_string.encode('utf-8'))

    st.download_button(
        label="Download JSON File",
        data=json_bytes, # Use the BytesIO object
        file_name=filename,
        mime="application/json"
    )
    
    # --- Optional: Basic Visualization ---
    st.subheader("Basic Trajectory Visualization (Sample)")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(env_width / 80, env_height / 80)) # Adjust figsize based on env dims

        # Plot Environment Bounds
        ax.set_xlim(0, env_width)
        ax.set_ylim(0, env_height)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Actor Trajectories')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.grid(True, linestyle='--', alpha=0.6)

        # Plot Obstacles
        for obs_w, obs_h, obs_x, obs_y in generated_data['environment']['obs']:
            rect = plt.Rectangle((obs_x, obs_y), obs_w, obs_h, color='gray', alpha=0.7)
            ax.add_patch(rect)

        # Plot Actor Trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, len(generated_data['actor']))) # Color map for actors
        
        for i, (actor_id, actor_data) in enumerate(generated_data['actor'].items()):
            x_coords = [frame[2] + frame[0]/2 for frame in actor_data['frames']] # Center point x
            y_coords = [frame[3] + frame[1]/2 for frame in actor_data['frames']] # Center point y
            
            if x_coords and y_coords: # Only plot if actor has frames
                ax.plot(x_coords, y_coords, marker='.', linestyle='-', markersize=2, label=f'Actor {actor_id}', color=colors[i])
                # Mark start and end points
                ax.plot(x_coords[0], y_coords[0], 'go', markersize=5, label=f'_Start {actor_id}') # Green circle for start
                ax.plot(x_coords[-1], y_coords[-1], 'rx', markersize=5, label=f'_End {actor_id}') # Red x for end

        # Improve legend if many actors
        if len(generated_data['actor']) <= 15:
             ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
             st.caption("Legend omitted for large number of actors.")
             
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free memory

    except ImportError:
        st.warning("Matplotlib not installed. Skipping visualization. Install with: pip install matplotlib")
    except Exception as e:
        st.error(f"An error occurred during visualization: {e}")


else:
    st.info("Configure parameters in the sidebar and click 'Generate Synthetic Data'.")