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

    # Check if close enough to target to avoid division by zero or tiny steps
    # Use speed as the threshold for simplicity
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
    # Check collision with slightly inflated proposed box for robustness? No, keep simple for now.
    proposed_box = (int(move_x), int(move_y), current_w, current_h)
    for obs_w, obs_h, obs_x, obs_y in obstacles:
        if is_colliding(proposed_box[0], proposed_box[1], proposed_box[2], proposed_box[3],
                        obs_x, obs_y, obs_w, obs_h):
            collision_obstacle = True
            break # Collision detected with one obstacle is enough

    if collision_boundary or collision_obstacle:
        # Simple collision response: Signal to pick a new random target
        # This prevents getting stuck directly against an obstacle/boundary
         return None, None

    return int(move_x), int(move_y)


def generate_synthetic_data(config):
    """Generates the synthetic data based on configuration."""

    env_w = config['env_dims'][0]
    env_h = config['env_dims'][1]
    num_actors = config['num_actors']
    duration_seconds = config['duration_seconds']
    framerate = config['framerate'] # Simulation rate
    sampling_rate = config['sampling_rate'] # Rate reported in JSON
    num_obstacles = config['num_obstacles']
    min_actor_w = config['min_actor_w']
    max_actor_w = config['max_actor_w']
    min_actor_h = config['min_actor_h']
    max_actor_h = config['max_actor_h']
    min_speed = config['min_speed']
    max_speed = config['max_speed']

    total_frames = duration_seconds * framerate # Simulation frames

    # --- Generate Environment ---
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(seconds=duration_seconds)

    obstacles = [generate_random_obstacle(env_w, env_h) for _ in range(num_obstacles)]

    environment_data = {
        'dims': (env_w, env_h),
        'obs': obstacles,
        'framerate': framerate, # Report simulation framerate
        'sampling': sampling_rate, # Report user-defined sampling rate
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
        start_x, start_y = -1, -1 # Initialize invalid
        attempts = 0
        max_attempts = 200 # Prevent infinite loop if space is too crowded
        while attempts < max_attempts:
            temp_x = random.randint(0, env_w - actor_w)
            temp_y = random.randint(0, env_h - actor_h)
            valid_start = True
            for obs_w, obs_h, obs_x, obs_y in obstacles:
                 if is_colliding(temp_x, temp_y, actor_w, actor_h, obs_x, obs_y, obs_w, obs_h):
                     valid_start = False
                     break
            if valid_start:
                start_x, start_y = temp_x, temp_y
                break
            attempts += 1
        
        if start_x == -1: # Fallback if no valid spot found
            print(f"Warning: Could not find obstacle-free start for actor {actor_id}. Placing at (0,0).")
            start_x, start_y = 0, 0


        # Random entry frame
        # Ensure actor can exist for at least 1 frame after entry for exit calculation
        max_possible_entry = max(0, total_frames - 1) # Can enter on the very last frame technically

        # Ensure entry frame calculation is valid if max_possible_entry is 0
        if max_possible_entry == 0:
            entry_frame = 0
        else:
            entry_frame = random.randint(0, max_possible_entry)


        # --- Corrected Duration and Exit Frame Calculation ---
        min_duration_frames = framerate * 2 # Minimum frames lifespan (e.g., 2 seconds)
        min_duration_frames = max(1, min_duration_frames) # Ensure minimum duration is at least 1 frame

        # Calculate max possible duration from entry frame until the end of simulation
        max_possible_duration = total_frames - entry_frame
        max_possible_duration = max(1, max_possible_duration) # Must exist for at least one frame

        # Determine the actual duration
        if max_possible_duration < min_duration_frames:
            actual_duration_frames = max_possible_duration
        else:
            actual_duration_frames = random.randint(min_duration_frames, max_possible_duration)

        # Calculate exit frame (inclusive index of the last frame the actor exists)
        exit_frame = entry_frame + actual_duration_frames - 1
        exit_frame = min(exit_frame, total_frames - 1) # Clamp to last simulation frame
        exit_frame = max(entry_frame, exit_frame) # Ensure exit >= entry
        # --- End of Corrected Section ---


        actors[actor_id] = {
            'descriptor': [f'person_{actor_id}'],
            'entry': [entry_frame],
            'exit': [exit_frame],
            'frames': [],
            'postures': [],
        }

        actor_states[actor_id] = {
            'id': actor_id,
            'w': actor_w,
            'h': actor_h,
            'x': start_x,
            'y': start_y,
            'speed': random.uniform(min_speed, max_speed),
            'target_x': random.randint(0, env_w - actor_w), # Initial target
            'target_y': random.randint(0, env_h - actor_h), # Initial target
            'entry_frame': entry_frame,
            'exit_frame': exit_frame,
            'active': False
        }
        # Ensure initial target is valid (not inside obstacle)
        attempts = 0
        while attempts < 50:
             is_clear = True
             for obs_w, obs_h, obs_x, obs_y in obstacles:
                 if is_colliding(actor_states[actor_id]['target_x'], actor_states[actor_id]['target_y'], actor_w, actor_h, obs_x, obs_y, obs_w, obs_h):
                     is_clear = False
                     actor_states[actor_id]['target_x'] = random.randint(0, env_w - actor_w)
                     actor_states[actor_id]['target_y'] = random.randint(0, env_h - actor_h)
                     break
             if is_clear:
                 break
             attempts +=1
        if not is_clear:
            print(f"Warning: Could not find obstacle-free initial target for actor {actor_id}.")
            # Keep potentially invalid target, movement logic should handle collision


    # --- Simulation Loop ---
    for frame_idx in range(total_frames):
        for actor_id, state in actor_states.items():

            # Determine if actor is active in this frame
            is_currently_active = (frame_idx >= state['entry_frame'] and frame_idx <= state['exit_frame'])

            if not state['active'] and is_currently_active:
                 # Actor becomes active NOW
                 state['active'] = True
                 # Position might need validation if entry_frame > 0 and obstacles exist
                 if frame_idx > 0: # Don't need to re-validate if starting at frame 0
                    is_clear = True
                    for obs_w, obs_h, obs_x, obs_y in obstacles:
                        if is_colliding(state['x'], state['y'], state['w'], state['h'], obs_x, obs_y, obs_w, obs_h):
                            is_clear = False
                            break
                    if not is_clear:
                        # Invalid starting position due to late entry, try to find a new spot
                        valid_pos = False
                        attempts = 0
                        while not valid_pos and attempts < 100:
                            temp_x = random.randint(0, env_w - state['w'])
                            temp_y = random.randint(0, env_h - state['h'])
                            is_clear_new = True
                            for obs_w, obs_h, obs_x, obs_y in obstacles:
                                if is_colliding(temp_x, temp_y, state['w'], state['h'], obs_x, obs_y, obs_w, obs_h):
                                    is_clear_new = False
                                    break
                            if is_clear_new:
                                state['x'] = temp_x
                                state['y'] = temp_y
                                valid_pos = True
                            attempts += 1
                        if not valid_pos: # Fallback
                             state['x'], state['y'] = 0, 0
                             print(f"Warning: Could not find valid re-entry spot for actor {actor_id} at frame {frame_idx}.")


            elif state['active'] and not is_currently_active:
                 # Actor becomes inactive NOW (should technically happen after the frame is processed)
                 state['active'] = False
                 # We already processed the last active frame in the previous iteration.
                 continue # Skip processing for this inactive frame


            if not state['active']:
                 # Actor is not active in this frame index range
                 continue # Skip to next actor


            # --- Actor Movement (only if active) ---
            current_x, current_y = state['x'], state['y']
            target_x, target_y = state['target_x'], state['target_y']

            next_x, next_y = get_next_position(current_x, current_y, state['w'], state['h'],
                                               target_x, target_y, state['speed'],
                                               env_w, env_h, obstacles)

            needs_new_target = (next_x is None or next_y is None)

            if needs_new_target:
                # Reached target or collided, pick new target
                valid_target = False
                attempts = 0
                new_target_x, new_target_y = -1, -1
                while not valid_target and attempts < 50: # Avoid infinite loop
                    temp_target_x = random.randint(0, env_w - state['w'])
                    temp_target_y = random.randint(0, env_h - state['h'])
                    is_clear = True
                    # Check if target *area* overlaps obstacle
                    for obs_w, obs_h, obs_x, obs_y in obstacles:
                        if is_colliding(temp_target_x, temp_target_y, state['w'], state['h'], obs_x, obs_y, obs_w, obs_h):
                            is_clear = False
                            break
                    if is_clear:
                         new_target_x, new_target_y = temp_target_x, temp_target_y
                         valid_target = True
                    attempts += 1

                if valid_target:
                    state['target_x'] = new_target_x
                    state['target_y'] = new_target_y
                else:
                     # Failed to find valid target, try moving away slightly? Or just stay put.
                     # Staying put is simplest for now.
                     print(f"Warning: Actor {actor_id} failed to find obstacle-free target, staying put.")
                     pass # Keep old target, maybe it becomes valid later?

                # Stay in current position for this frame if no valid *move* could be calculated
                # or if we picked a new target this frame
                next_x, next_y = current_x, current_y


            # Update state position
            state['x'] = next_x
            state['y'] = next_y

            # Record frame data for the actor
            frame_data = (state['w'], state['h'], state['x'], state['y'])
            actors[actor_id]['frames'].append(frame_data)
            actors[actor_id]['postures'].append(0) # Placeholder posture '0'


    # --- Final Data Structure ---
    # Clean up actors with no frames (shouldn't happen with corrected logic, but good practice)
    final_actors = {aid: data for aid, data in actors.items() if data['frames']}

    # Adjust entry/exit frame numbers based on actual recorded frames if needed
    # For now, we trust the initial calculated entry/exit frame numbers.
    # If an actor has frames, its entry/exit should be valid.

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

# Initialize session state for generated data if it doesn't exist
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None

# --- Configuration Sidebar ---
st.sidebar.header("Configuration")

# Environment Settings
st.sidebar.subheader("Environment Settings")
env_width = st.sidebar.number_input("Environment Width", min_value=50, value=800, step=50)
env_height = st.sidebar.number_input("Environment Height", min_value=50, value=600, step=50)
num_obstacles = st.sidebar.slider("Number of Random Obstacles", min_value=0, max_value=20, value=5)
framerate = st.sidebar.slider("Simulation Framerate (fps)", min_value=1, max_value=60, value=15, help="Internal simulation steps per second.")
# --- NEW SAMPLING RATE SLIDER ---
sampling_rate = st.sidebar.slider("Sampling Rate (fps)", min_value=1, max_value=60, value=15, help="Rate at which data is 'recorded' or outputted in the JSON.")
duration_seconds = st.sidebar.number_input("Simulation Duration (seconds)", min_value=5, max_value=600, value=30, step=5)
# chunk_time = st.sidebar.number_input("Chunk Time (seconds)", min_value=10, value=600) # Less critical for generation

# Actor Settings
st.sidebar.subheader("Actor Settings")
num_actors = st.sidebar.slider("Number of Actors", min_value=1, max_value=50, value=10)
min_actor_w = st.sidebar.slider("Min Actor Width", min_value=5, max_value=50, value=15)
max_actor_w = st.sidebar.slider("Max Actor Width", min_value=5, max_value=80, value=30)
min_actor_h = st.sidebar.slider("Min Actor Height", min_value=5, max_value=50, value=15)
max_actor_h = st.sidebar.slider("Max Actor Height", min_value=5, max_value=80, value=30)
# --- INCREASED SPEED LIMITS ---
min_speed = st.sidebar.slider("Min Actor Speed (units/frame)", min_value=0.1, max_value=20.0, value=0.5, step=0.1)
max_speed = st.sidebar.slider("Max Actor Speed (units/frame)", min_value=0.5, max_value=50.0, value=3.0, step=0.5) # Increased max and default, adjusted step

# Ensure min <= max sliders logic
if min_actor_w > max_actor_w: max_actor_w = min_actor_w
if min_actor_h > max_actor_h: max_actor_h = min_actor_h
if min_speed > max_speed: max_speed = min_speed

# --- Generate Button ---
st.header("Generate Data")

if st.button("Generate Synthetic Data"):

    config = {
        'env_dims': (env_width, env_height),
        'num_obstacles': num_obstacles,
        'framerate': framerate,         # Simulation rate
        'sampling_rate': sampling_rate, # Added sampling rate
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
        # Generate data and store it in session state
        st.session_state.generated_data = generate_synthetic_data(config)

    st.success("Data Generation Complete!")
    # Note: The rest of the display/download logic is now outside this button's block,
    # relying on st.session_state.generated_data

# --- Display Results and Download (if data exists in session state) ---
if st.session_state.generated_data is not None:
    generated_data = st.session_state.generated_data # Retrieve from session state

    # Display JSON output
    st.subheader("Generated JSON Data")
    st.json(generated_data, expanded=False) # Show collapsed by default

    # Provide download button
    st.subheader("Download Data")
    try:
        # Use indent=4 for readability in downloaded file
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
    except Exception as e:
        st.error(f"Error preparing download: {e}")


    # --- Optional: Basic Visualization ---
    st.subheader("Basic Trajectory Visualization (Sample)")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(5, env_width / 100), max(5, env_height / 100))) # Adjust figsize based on env dims, ensure min size

        # Plot Environment Bounds
        ax.set_xlim(0, env_width)
        ax.set_ylim(0, env_height)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Actor Trajectories')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.invert_yaxis() # Often Y=0 is top in image coordinates
        ax.grid(True, linestyle='--', alpha=0.6)

        # Plot Obstacles
        if 'obs' in generated_data['environment']:
            for obs_w, obs_h, obs_x, obs_y in generated_data['environment']['obs']:
                rect = plt.Rectangle((obs_x, obs_y), obs_w, obs_h, color='gray', alpha=0.7, label='_Obstacle') # Use _nolabel_
                ax.add_patch(rect)

        # Plot Actor Trajectories
        if 'actor' in generated_data and generated_data['actor']: # Check if actor dict exists and is not empty
            num_plot_actors = len(generated_data['actor'])
            colors = plt.cm.viridis(np.linspace(0, 1, num_plot_actors)) # Color map for actors

            for i, (actor_id, actor_data) in enumerate(generated_data['actor'].items()):
                 # Check if 'frames' data is present and not empty
                 if 'frames' in actor_data and actor_data['frames']:
                     x_coords = [frame[2] + frame[0]/2 for frame in actor_data['frames']] # Center point x
                     y_coords = [frame[3] + frame[1]/2 for frame in actor_data['frames']] # Center point y

                     if x_coords and y_coords: # Ensure lists are not empty after extraction
                         ax.plot(x_coords, y_coords, marker='.', linestyle='-', markersize=2, label=f'Actor {actor_id}', color=colors[i % len(colors)]) # Use modulo for color safety
                         # Mark start and end points
                         ax.plot(x_coords[0], y_coords[0], 'go', markersize=5, label=f'_Start {actor_id}') # Green circle for start
                         ax.plot(x_coords[-1], y_coords[-1], 'rx', markersize=5, label=f'_End {actor_id}') # Red x for end
                 else:
                     st.caption(f"Actor {actor_id} has no frame data to plot.")


            # Improve legend if many actors
            handles, labels = ax.get_legend_handles_labels()
            # Filter out obstacle labels if needed (though _nolabel_ should handle it)
            filtered_labels_handles = [(h, l) for h, l in zip(handles, labels) if not l.startswith('_')]
            
            if len(filtered_labels_handles) > 0:
                 if len(filtered_labels_handles) <= 15:
                      # Place legend outside plot
                      ax.legend(handles=[h for h, l in filtered_labels_handles], 
                                labels=[l for h, l in filtered_labels_handles], 
                                bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                 else:
                      st.caption("Legend omitted for large number of actors.")
            else:
                 st.caption("No actor trajectories to include in legend.")


        st.pyplot(fig)
        plt.close(fig) # Close the figure to free memory

    except ImportError:
        st.warning("Matplotlib not installed. Skipping visualization. Install with: pip install matplotlib")
    except Exception as e:
        st.error(f"An error occurred during visualization: {e}")
        st.exception(e) # Show full traceback for debugging


# Show message if no data has been generated yet
elif not st.session_state.generated_data:
    st.info("Configure parameters in the sidebar and click 'Generate Synthetic Data'.")