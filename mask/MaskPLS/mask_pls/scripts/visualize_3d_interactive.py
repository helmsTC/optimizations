"""
Interactive 3D visualization for point cloud segmentation results using OpenCV
Allows rotation, zoom, and pan with mouse and keyboard controls
"""

import numpy as np
import cv2
import click
from pathlib import Path
import json
import math


class Interactive3DVisualizer:
    """Interactive 3D point cloud visualizer with OpenCV"""
    
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        
        # Camera parameters
        self.camera_distance = 50.0
        self.camera_pitch = -30.0  # Up/down rotation
        self.camera_yaw = 45.0     # Left/right rotation
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_z = 0.0
        self.fov = 60.0
        
        # Mouse interaction state
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_left_pressed = False
        self.mouse_right_pressed = False
        self.mouse_middle_pressed = False
        
        # Display options
        self.point_size = 2
        self.show_axes = True
        self.show_grid = True
        self.show_stats = True
        self.current_view = 'instance'  # 'semantic', 'instance', 'height', 'intensity'
        
        # Color maps
        self.semantic_colors = {
            0: [0, 0, 0],           # unlabeled
            1: [255, 0, 0],         # car
            2: [255, 100, 100],     # bicycle
            3: [255, 200, 100],     # motorcycle
            4: [255, 255, 0],       # truck
            5: [100, 255, 100],     # other-vehicle
            6: [0, 255, 0],         # person
            7: [0, 100, 255],       # bicyclist
            8: [0, 0, 255],         # motorcyclist
            9: [255, 0, 255],       # road
            10: [200, 200, 200],    # parking
            11: [150, 150, 150],    # sidewalk
            12: [100, 100, 100],    # other-ground
            13: [0, 255, 255],      # building
            14: [255, 128, 0],      # fence
            15: [128, 0, 255],      # vegetation
            16: [255, 255, 128],    # trunk
            17: [128, 255, 128],    # terrain
            18: [128, 128, 255],    # pole
            19: [255, 128, 255],    # traffic-sign
        }
        
        # Generate instance colors
        np.random.seed(42)
        self.instance_colors = {}
        for i in range(1000):
            self.instance_colors[i] = [
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            ]
        self.instance_colors[0] = [0, 0, 0]
        
    def load_results(self, results_dir, filename_base):
        """Load saved results"""
        results_dir = Path(results_dir)
        
        # Load colored point cloud
        colored_path = results_dir / f"{filename_base}_colored.npy"
        if colored_path.exists():
            colored_points = np.load(colored_path)
            self.points = colored_points[:, :3]
            self.intensity = colored_points[:, 3] if colored_points.shape[1] > 3 else np.ones(len(colored_points))
            self.semantic_labels = colored_points[:, 4].astype(int) if colored_points.shape[1] > 4 else np.zeros(len(colored_points), dtype=int)
            self.instance_labels = colored_points[:, 5].astype(int) if colored_points.shape[1] > 5 else np.zeros(len(colored_points), dtype=int)
        else:
            raise FileNotFoundError(f"Colored point cloud not found: {colored_path}")
        
        # Load object detections
        objects_path = results_dir / f"{filename_base}_objects.json"
        if objects_path.exists():
            with open(objects_path, 'r') as f:
                self.detected_objects = json.load(f)
        else:
            self.detected_objects = []
        
        # Compute bounds
        self.bounds_min = np.min(self.points, axis=0)
        self.bounds_max = np.max(self.points, axis=0)
        self.center = (self.bounds_min + self.bounds_max) / 2
        
        # Center points
        self.points_centered = self.points - self.center
        
        print(f"Loaded {len(self.points)} points")
        print(f"Bounds: [{self.bounds_min[0]:.1f}, {self.bounds_max[0]:.1f}] x "
              f"[{self.bounds_min[1]:.1f}, {self.bounds_max[1]:.1f}] x "
              f"[{self.bounds_min[2]:.1f}, {self.bounds_max[2]:.1f}]")
        print(f"Detected objects: {len(self.detected_objects)}")
    
    def project_points(self, points):
        """Project 3D points to 2D screen coordinates"""
        # Create rotation matrices
        pitch_rad = math.radians(self.camera_pitch)
        yaw_rad = math.radians(self.camera_yaw)
        
        # Rotation matrix around X axis (pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)]
        ])
        
        # Rotation matrix around Y axis (yaw)
        Ry = np.array([
            [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
            [0, 1, 0],
            [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]
        ])
        
        # Apply rotations
        rotated = points @ Rx.T @ Ry.T
        
        # Translate by camera position
        translated = rotated + np.array([self.camera_x, self.camera_y, self.camera_z - self.camera_distance])
        
        # Perspective projection
        focal_length = self.width / (2 * math.tan(math.radians(self.fov / 2)))
        
        # Avoid division by zero
        z_clip = np.maximum(translated[:, 2], 0.1)
        
        # Project to screen
        screen_x = (translated[:, 0] * focal_length / z_clip) + self.width / 2
        screen_y = (-translated[:, 1] * focal_length / z_clip) + self.height / 2
        
        # Create screen points
        screen_points = np.column_stack([screen_x, screen_y])
        
        # Depth for sorting (farther points drawn first)
        depths = translated[:, 2]
        
        return screen_points.astype(int), depths
    
    def get_point_colors(self):
        """Get colors based on current view mode"""
        if self.current_view == 'semantic':
            colors = np.array([self.semantic_colors.get(label, [128, 128, 128]) 
                              for label in self.semantic_labels])
        elif self.current_view == 'instance':
            colors = np.array([self.instance_colors.get(label, [128, 128, 128]) 
                              for label in self.instance_labels])
        elif self.current_view == 'height':
            # Color by height
            heights = self.points[:, 2]
            normalized = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
            colors = plt.cm.jet(normalized)[:, :3] * 255
            colors = colors.astype(int)
        else:  # intensity
            normalized = (self.intensity - self.intensity.min()) / (self.intensity.max() - self.intensity.min() + 1e-6)
            colors = plt.cm.viridis(normalized)[:, :3] * 255
            colors = colors.astype(int)
        
        return colors
    
    def draw_axes(self, img):
        """Draw 3D axes"""
        # Define axis endpoints
        axis_length = 10.0
        axes = np.array([
            [[0, 0, 0], [axis_length, 0, 0]],  # X axis - red
            [[0, 0, 0], [0, axis_length, 0]],  # Y axis - green
            [[0, 0, 0], [0, 0, axis_length]]   # Z axis - blue
        ])
        
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
        
        for axis, color in zip(axes, colors):
            points_2d, _ = self.project_points(axis)
            if len(points_2d) == 2:
                cv2.line(img, tuple(points_2d[0]), tuple(points_2d[1]), color, 2)
                cv2.putText(img, ['X', 'Y', 'Z'][colors.index(color)], 
                           tuple(points_2d[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
    
    def draw_grid(self, img):
        """Draw ground grid"""
        grid_size = 10
        grid_count = 5
        
        # Create grid lines
        lines = []
        for i in range(-grid_count, grid_count + 1):
            # Lines parallel to X axis
            lines.append([[-grid_size * grid_count, i * grid_size, 0],
                         [grid_size * grid_count, i * grid_size, 0]])
            # Lines parallel to Y axis
            lines.append([[i * grid_size, -grid_size * grid_count, 0],
                         [i * grid_size, grid_size * grid_count, 0]])
        
        # Project and draw lines
        for line in lines:
            points_2d, _ = self.project_points(np.array(line))
            if len(points_2d) == 2:
                cv2.line(img, tuple(points_2d[0]), tuple(points_2d[1]), (100, 100, 100), 1)
    
    def draw_stats(self, img):
        """Draw statistics and controls"""
        # Background for text
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # Draw text
        y_offset = 30
        texts = [
            f"Points: {len(self.points)}",
            f"View: {self.current_view.capitalize()}",
            f"Objects: {len(self.detected_objects)}",
            f"Camera: ({self.camera_yaw:.0f}°, {self.camera_pitch:.0f}°)",
            f"Distance: {self.camera_distance:.1f}m",
            "",
            "Controls:",
            "  LMB + Drag: Rotate",
            "  RMB + Drag: Pan",
            "  Scroll: Zoom",
            "  1-4: Change view",
            "  +/-: Point size",
            "  G: Toggle grid",
            "  A: Toggle axes",
            "  R: Reset view",
            "  Q: Quit"
        ]
        
        for text in texts:
            cv2.putText(img, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            y_offset += 20
    
    def render(self):
        """Render the point cloud"""
        # Create black image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw grid
        if self.show_grid:
            self.draw_grid(img)
        
        # Project points
        screen_points, depths = self.project_points(self.points_centered)
        
        # Get colors
        colors = self.get_point_colors()
        
        # Sort by depth (draw far points first)
        sort_indices = np.argsort(-depths)
        
        # Draw points
        for idx in sort_indices:
            pt = screen_points[idx]
            if 0 <= pt[0] < self.width and 0 <= pt[1] < self.height:
                color = colors[idx]
                cv2.circle(img, tuple(pt), self.point_size, color[::-1].tolist(), -1)
        
        # Draw axes
        if self.show_axes:
            self.draw_axes(img)
        
        # Draw stats
        if self.show_stats:
            self.draw_stats(img)
        
        return img
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_left_pressed = True
            self.mouse_last_x = x
            self.mouse_last_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_left_pressed = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_right_pressed = True
            self.mouse_last_x = x
            self.mouse_last_y = y
        elif event == cv2.EVENT_RBUTTONUP:
            self.mouse_right_pressed = False
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.mouse_middle_pressed = True
            self.mouse_last_x = x
            self.mouse_last_y = y
        elif event == cv2.EVENT_MBUTTONUP:
            self.mouse_middle_pressed = False
        elif event == cv2.EVENT_MOUSEMOVE:
            dx = x - self.mouse_last_x
            dy = y - self.mouse_last_y
            
            if self.mouse_left_pressed:
                # Rotate camera
                self.camera_yaw += dx * 0.5
                self.camera_pitch -= dy * 0.5
                self.camera_pitch = np.clip(self.camera_pitch, -89, 89)
            elif self.mouse_right_pressed:
                # Pan camera
                pan_speed = self.camera_distance * 0.001
                self.camera_x -= dx * pan_speed
                self.camera_y += dy * pan_speed
            
            self.mouse_last_x = x
            self.mouse_last_y = y
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom
            if flags > 0:
                self.camera_distance *= 0.9
            else:
                self.camera_distance *= 1.1
            self.camera_distance = np.clip(self.camera_distance, 5, 200)
    
    def reset_view(self):
        """Reset camera to default view"""
        self.camera_distance = 50.0
        self.camera_pitch = -30.0
        self.camera_yaw = 45.0
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_z = 0.0
    
    def run(self, results_dir, filename_base):
        """Main visualization loop"""
        # Load data
        self.load_results(results_dir, filename_base)
        
        # Import matplotlib for colormaps
        global plt
        import matplotlib.pyplot as plt
        
        # Create window
        window_name = "3D Point Cloud Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\nInteractive 3D Viewer Started")
        print("Use mouse and keyboard to navigate (see overlay for controls)")
        
        while True:
            # Render frame
            img = self.render()
            
            # Display
            cv2.imshow(window_name, img)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('1'):
                self.current_view = 'semantic'
            elif key == ord('2'):
                self.current_view = 'instance'
            elif key == ord('3'):
                self.current_view = 'height'
            elif key == ord('4'):
                self.current_view = 'intensity'
            elif key == ord('g'):
                self.show_grid = not self.show_grid
            elif key == ord('a'):
                self.show_axes = not self.show_axes
            elif key == ord('s'):
                self.show_stats = not self.show_stats
            elif key == ord('r'):
                self.reset_view()
            elif key == ord('+') or key == ord('='):
                self.point_size = min(self.point_size + 1, 10)
            elif key == ord('-'):
                self.point_size = max(self.point_size - 1, 1)
            elif key == ord('p'):
                # Save screenshot
                cv2.imwrite(f"{filename_base}_screenshot.png", img)
                print(f"Screenshot saved to {filename_base}_screenshot.png")
        
        cv2.destroyAllWindows()


@click.command()
@click.argument('results_dir', type=click.Path(exists=True))
@click.argument('filename_base')
@click.option('--width', default=1200, help='Window width')
@click.option('--height', default=800, help='Window height')
def main(results_dir, filename_base, width, height):
    """
    Interactive 3D visualization of point cloud segmentation results
    
    Examples:
        python visualize_3d_interactive.py results/ pointcloud_000000
        python visualize_3d_interactive.py results/ pointcloud_000000 --width 1600 --height 900
    """
    visualizer = Interactive3DVisualizer(width, height)
    visualizer.run(results_dir, filename_base)


if __name__ == "__main__":
    main()