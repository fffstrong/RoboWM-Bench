import numpy as np
import plotly.graph_objects as go
import sys

# ==========================================
# Configuration: Please replace with actual file path
# ==========================================
file_path = 'data/processed/hand_dataset/cook_bi/1/1_human_left_black/smoothing_processor/smoothed_actions_left_single_arm.npz' 

try:
    # ==========================================
    # 1. Load data & display Keys
    # ==========================================
    print(f"Loading file: {file_path} ...")
    try:
        data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        sys.exit(1)

    print("\n" + "="*50)
    print(f"{'Key Name':<30} | {'Shape':<20}")
    print("-" * 50)
    
    keys = data.files if hasattr(data, 'files') else list(data.keys())
    
    for key in keys:
        try:
            shape_str = str(data[key].shape)
            print(f"{key:<30} | {shape_str:<20}")
        except:
            print(f"{key:<30} | {'(Unknown/Scalar)':<20}")
    print("="*50 + "\n")

    # ==========================================
    # 2. User interaction selection
    # ==========================================
    target_key = input("Please enter the Key Name to visualize (e.g. kpts_3d): ").strip()

    if target_key not in data:
        print(f"Error: Key '{target_key}' is not in the file.")
        sys.exit(1)

    kpts_data = data[target_key]

    # Data validation
    if kpts_data.ndim != 3:
        print(f"Error: Data dimensions {kpts_data.shape} do not match the (Time, Points, Dim) temporal format.")
        sys.exit(1)

    num_frames, num_points, dim = kpts_data.shape
    print(f"\nSelected data: {target_key}")
    print(f"Frames: {num_frames}, Points: {num_points}, Dimensions: {dim}D")

    # ==========================================
    # 3. Prepare Plotly animation data
    # ==========================================
    frames = []
    initial_data = kpts_data[0] 

    # ------------------------------------------
    # Branch A: 3D Visualization (Add uirevision and fixed Camera)
    # ------------------------------------------
    if dim == 3:
        print("3D data detected, building 3D scene...")
        
        # 1. Calculate global fixed range
        x_min, x_max = np.min(kpts_data[:, :, 0]), np.max(kpts_data[:, :, 0])
        y_min, y_max = np.min(kpts_data[:, :, 1]), np.max(kpts_data[:, :, 1])
        z_min, z_max = np.min(kpts_data[:, :, 2]), np.max(kpts_data[:, :, 2])

        pad_ratio = 0.1 # Slightly increase margin
        x_pad = (x_max - x_min) * pad_ratio
        y_pad = (y_max - y_min) * pad_ratio
        z_pad = (z_max - z_min) * pad_ratio
        
        def create_trace_3d(frame_data):
            return go.Scatter3d(
                x=frame_data[:, 0],
                y=frame_data[:, 1],
                z=frame_data[:, 2],
                mode='markers+text',
                text=[str(i) for i in range(num_points)],
                textposition="top center",
                marker=dict(size=4, color='red'),
                name='Keypoints'
            )

        fig = go.Figure(data=[create_trace_3d(initial_data)])

        for i in range(num_frames):
            frames.append(go.Frame(
                data=[create_trace_3d(kpts_data[i])],
                name=str(i)
            ))

        layout_settings = dict(
            title=f"3D Visualization: {target_key}",
            scene=dict(
                xaxis=dict(title='X', range=[x_min - x_pad, x_max + x_pad], autorange=False),
                yaxis=dict(title='Y', range=[y_min - y_pad, y_max + y_pad], autorange=False),
                zaxis=dict(title='Z', range=[z_min - z_pad, z_max + z_pad], autorange=False),
                aspectmode='manual', # Force manual aspect ratio
                aspectratio=dict(x=1, y=1, z=1), # This ratio can be adjusted based on actual object proportions, or use 'data'
            )
        )

    # ------------------------------------------
    # Branch B: 2D Visualization
    # ------------------------------------------
    elif dim == 2:
        print("2D data detected, building 2D plane...")
        
        def create_trace_2d(frame_data):
            return go.Scatter(
                x=frame_data[:, 0],
                y=frame_data[:, 1],
                mode='markers+text',
                text=[str(i) for i in range(num_points)],
                textposition="top center",
                marker=dict(size=10, color='red'),
                name='Keypoints'
            )

        fig = go.Figure(data=[create_trace_2d(initial_data)])

        for i in range(num_frames):
            frames.append(go.Frame(
                data=[create_trace_2d(kpts_data[i])],
                name=str(i)
            ))

        layout_settings = dict(
            title=f"2D Visualization: {target_key}",
            xaxis=dict(title='X', autorange=True),
            yaxis=dict(title='Y', autorange='reversed', scaleanchor="x", scaleratio=1),
        )
    
    else:
        print(f"Unsupported dimension: {dim}.")
        sys.exit(1)

    # ==========================================
    # 4. Unified layout (Added uirevision)
    # ==========================================
    fig.frames = frames

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

    common_layout = dict(
        # [Core fix] uirevision: As long as this value remains unchanged, Plotly will not reset user interactions (zoom, rotation)
        # This prevents the Camera from being forced back to its original position during each frame playback
        uirevision='dataset_v1', 
        
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
        width=900, height=800
    )
    
    fig.update_layout(**layout_settings)
    fig.update_layout(**common_layout)

    # ==========================================
    # 5. Display
    # ==========================================
    print("Generation complete, opening browser...")
    fig.show()

except Exception as e:
    print(f"\nException occurred: {e}")
    import traceback
    traceback.print_exc()