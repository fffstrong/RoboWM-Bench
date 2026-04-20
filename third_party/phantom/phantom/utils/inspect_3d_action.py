import numpy as np
import plotly.graph_objects as go

file_path = 'data/processed/collected_data/1/smoothing_processor/smoothed_actions_left_single_arm.npz'

try:
    data = np.load(file_path, allow_pickle=True)
    
    # Print information (as requested)
    print(f"{'Key Name':<25} | {'Shape':<15}")
    print("-" * 45)
    for key in data.files:
        print(f"{key:<25} | {str(data[key].shape):<15}")

    # Prepare Plotly canvas
    fig = go.Figure()

    # Extract and add trajectories
    arms = {
        'Left Arm': ('ee_pts', 'blue'),
        # 'Right Arm': ('action_pos_right', 'orange')
    }

    for arm_name, (key, color) in arms.items():
        if key in data:
            traj = data[key]
            x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
            
            # Add 3D trajectory line
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                name=f'{arm_name} Trajectory',
                line=dict(color=color, width=4)
            ))
            
            # Add start point (green dot)
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers',
                name=f'{arm_name} Start',
                marker=dict(size=8, color='green')
            ))

            # Add end point (red square)
            fig.add_trace(go.Scatter3d(
                x=[x[-1]], y=[y[-1]], z=[z[-1]],
                mode='markers',
                name=f'{arm_name} End',
                marker=dict(size=8, color='red', symbol='square')
            ))

    # Set layout: fixed aspect ratio so the space doesn't look distorted
    fig.update_layout(
        title="Interactive Robot Arm Trajectory",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            aspectmode='data' # Important: maintain true 1:1:1 scale
        ),
        margin=dict(l=0, r=0, b=0, t=40) # Reduce blank margins
    )

    # Show chart (will open automatically in the browser)
    fig.show()

except Exception as e:
    print(f"Error: {e}")
