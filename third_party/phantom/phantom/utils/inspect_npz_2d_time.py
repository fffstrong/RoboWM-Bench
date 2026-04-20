import numpy as np
import plotly.graph_objects as go

# Replace with your actual file path
file_path = 'data/processed/collected_data/1/hand_processor/hand_data_left.npz'

try:
    # ==========================================
    # 1. Load data
    # ==========================================
    # For demonstration, if the file doesn't exist, generate random data matching your description
    # In actual use, please ensure the file path is correct, the code will read it automatically
    try:
        data = np.load(file_path, allow_pickle=True)
        kpts_2d = data['kpts_2d']
        print(f"Loaded from {file_path}")
    except FileNotFoundError:
        print("File not found, generating demo data (Shape: 184, 21, 2)...")
        # Simulate data generation
        frames, num_kpts = 184, 21
        kpts_2d = np.zeros((frames, num_kpts, 2))
        kpts_2d[0] = np.random.uniform(123, 829, size=(num_kpts, 2))
        for i in range(1, frames):
            kpts_2d[i] = kpts_2d[i-1] + np.random.normal(0, 10, size=(num_kpts, 2))
        # Simulate data dictionary
        data = {'kpts_2d': kpts_2d}

    # ==========================================
    # 2. Print information (keeping your style)
    # ==========================================
    print(f"{'Key Name':<25} | {'Shape':<15}")
    print("-" * 45)
    for key in data.keys(): # .npz files use .files, dictionaries use .keys()
        shape_str = str(data[key].shape)
        print(f"{key:<25} | {shape_str:<15}")

    # Get core data
    # kpts_2d shape: (Time, Points, 2) -> (184, 21, 2)
    kpts_data = data['kpts_2d'] 
    num_frames = kpts_data.shape[0]
    num_points = kpts_data.shape[1]

    # ==========================================
    # 3. Prepare Plotly animation data
    # ==========================================
    # Create the trajectory for the initial frame (Frame 0)
    # Note: Your data is 2D, so use go.Scatter instead of go.Scatter3d
    
    # Define connecting lines (Skeleton) - If specific bone connections are unknown, lines are drawn in point index order for demonstration
    # If lines are not needed, change mode to 'markers'
    initial_x = kpts_data[0, :, 0]
    initial_y = kpts_data[0, :, 1] # In image coordinates, Y usually points down, Plotly defaults to up, layout will invert it later

    fig = go.Figure(
        data=[
            go.Scatter(
                x=initial_x, 
                y=initial_y,
                mode='markers+text', # Display points and indices
                name='Keypoints',
                text=[str(i) for i in range(num_points)], # Mark point indices 0-20
                textposition="top center",
                marker=dict(size=10, color='red'),
                line=dict(width=2, color='blue')
            )
        ]
    )

    # Create data for each frame (Frames)
    frames = []
    for i in range(num_frames):
        frames.append(go.Frame(
            data=[go.Scatter(
                x=kpts_data[i, :, 0],
                y=kpts_data[i, :, 1]
            )],
            name=str(i) # Frame name used for slider correspondence
        ))
    
    fig.frames = frames

    # ==========================================
    # 4. Set layout and interactive controls
    # ==========================================
    # Define slider (Slider)
    sliders = [dict(
        steps=[dict(
            method='animate',
            args=[[str(k)], dict(mode='immediate', frame=dict(duration=50, redraw=True), transition=dict(duration=0))],
            label=str(k)
        ) for k in range(num_frames)], 
        transition=dict(duration=0),
        x=0, y=0, 
        currentvalue=dict(font=dict(size=12), prefix='Frame: ', visible=True, xanchor='right'),
        len=1.0
    )]

    fig.update_layout(
        title=f"2D Keypoints Visualization ({num_frames} frames)",
        
        # Axis settings
        xaxis=dict(title='X', range=[0, 1000], autorange=False),
        yaxis=dict(title='Y', range=[1000, 0], autorange=False), # Invert Y axis to adapt to image coordinate system
        
        # Maintain aspect ratio (important even for 2D)
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        
        # Play button
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=1, x=1.1, xanchor='right', yanchor='top',
            pad=dict(t=0, r=10),
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0))]
            ),
            dict(
                label='Pause',
                method='animate',
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))]
            )]
        )],
        
        sliders=sliders,
        width=800, height=800 # Canvas size
    )

    # ==========================================
    # 5. Show chart
    # ==========================================
    print("Generating visualization, please view in browser...")
    fig.show()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()