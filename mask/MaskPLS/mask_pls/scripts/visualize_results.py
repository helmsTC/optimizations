"""
Visualization script for point cloud segmentation results using OpenCV
Displays semantic and instance segmentation results in 2D projections
"""

import numpy as np
import cv2
import click
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib import cm


class PointCloudVisualizer:
    """Visualize point cloud segmentation results using OpenCV"""
    
    def __init__(self):
        # Define color maps for semantic classes (KITTI)
        self.semantic_colors = {
            0: [0, 0, 0],           # unlabeled - black
            1: [255, 0, 0],         # car - red
            2: [255, 100, 100],     # bicycle - light red
            3: [255, 200, 100],     # motorcycle - orange
            4: [255, 255, 0],       # truck - yellow
            5: [100, 255, 100],     # other-vehicle - light green
            6: [0, 255, 0],         # person - green
            7: [0, 100, 255],       # bicyclist - light blue
            8: [0, 0, 255],         # motorcyclist - blue
            9: [255, 0, 255],       # road - magenta
            10: [200, 200, 200],    # parking - light gray
            11: [150, 150, 150],    # sidewalk - gray
            12: [100, 100, 100],    # other-ground - dark gray
            13: [0, 255, 255],      # building - cyan
            14: [255, 128, 0],      # fence - orange
            15: [128, 0, 255],      # vegetation - purple
            16: [255, 255, 128],    # trunk - light yellow
            17: [128, 255, 128],    # terrain - light green
            18: [128, 128, 255],    # pole - light purple
            19: [255, 128, 255],    # traffic-sign - pink
        }
        
        # Generate random colors for instances
        np.random.seed(42)
        self.instance_colors = {}
        for i in range(1000):
            self.instance_colors[i] = [
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            ]
        self.instance_colors[0] = [0, 0, 0]  # No instance - black
        
    def load_results(self, results_dir, filename_base):
        """Load saved results"""
        results_dir = Path(results_dir)
        
        # Load colored point cloud
        colored_path = results_dir / f"{filename_base}_colored.npy"
        if colored_path.exists():
            colored_points = np.load(colored_path)
            points = colored_points[:, :3]
            intensity = colored_points[:, 3] if colored_points.shape[1] > 3 else None
            semantic_labels = colored_points[:, 4].astype(int) if colored_points.shape[1] > 4 else None
            instance_labels = colored_points[:, 5].astype(int) if colored_points.shape[1] > 5 else None
        else:
            raise FileNotFoundError(f"Colored point cloud not found: {colored_path}")
        
        # Load object detections
        objects_path = results_dir / f"{filename_base}_objects.json"
        if objects_path.exists():
            with open(objects_path, 'r') as f:
                detected_objects = json.load(f)
        else:
            detected_objects = []
        
        return points, intensity, semantic_labels, instance_labels, detected_objects
    
    def project_to_bev(self, points, semantic_labels=None, instance_labels=None,
                       x_range=(-50, 50), y_range=(-50, 50), resolution=0.1):
        """
        Project point cloud to Bird's Eye View (BEV)
        
        Args:
            points: [N, 3] point cloud
            semantic_labels: [N] semantic class labels
            instance_labels: [N] instance IDs
            x_range: (min, max) range in X
            y_range: (min, max) range in Y
            resolution: meters per pixel
            
        Returns:
            bev_semantic: BEV image with semantic colors
            bev_instance: BEV image with instance colors
        """
        # Calculate image dimensions
        width = int((x_range[1] - x_range[0]) / resolution)
        height = int((y_range[1] - y_range[0]) / resolution)
        
        # Initialize BEV images
        bev_semantic = np.zeros((height, width, 3), dtype=np.uint8)
        bev_instance = np.zeros((height, width, 3), dtype=np.uint8)
        bev_height = np.full((height, width), -np.inf)
        
        # Filter points within range
        mask = (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1]) & \
               (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1])
        
        filtered_points = points[mask]
        filtered_semantic = semantic_labels[mask] if semantic_labels is not None else None
        filtered_instance = instance_labels[mask] if instance_labels is not None else None
        
        # Convert to pixel coordinates
        px = ((filtered_points[:, 0] - x_range[0]) / resolution).astype(int)
        py = ((filtered_points[:, 1] - y_range[0]) / resolution).astype(int)
        
        # Clip to image bounds
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        
        # Fill BEV images (keep highest point at each pixel)
        for i in range(len(filtered_points)):
            if filtered_points[i, 2] > bev_height[py[i], px[i]]:
                bev_height[py[i], px[i]] = filtered_points[i, 2]
                
                if filtered_semantic is not None:
                    color = self.semantic_colors.get(filtered_semantic[i], [128, 128, 128])
                    bev_semantic[py[i], px[i]] = color[::-1]  # BGR for OpenCV
                
                if filtered_instance is not None:
                    color = self.instance_colors.get(filtered_instance[i], [128, 128, 128])
                    bev_instance[py[i], px[i]] = color[::-1]  # BGR for OpenCV
        
        # Flip vertically to match typical BEV orientation
        bev_semantic = cv2.flip(bev_semantic, 0)
        bev_instance = cv2.flip(bev_instance, 0)
        
        return bev_semantic, bev_instance
    
    def project_to_range_image(self, points, semantic_labels=None, instance_labels=None,
                               fov_up=3.0, fov_down=-25.0, width=1024, height=64):
        """
        Project point cloud to range image (like LiDAR sensor view)
        
        Args:
            points: [N, 3] point cloud
            semantic_labels: [N] semantic class labels
            instance_labels: [N] instance IDs
            fov_up: Upper field of view in degrees
            fov_down: Lower field of view in degrees
            width: Image width (horizontal resolution)
            height: Image height (vertical resolution)
            
        Returns:
            range_semantic: Range image with semantic colors
            range_instance: Range image with instance colors
            range_depth: Range image with depth values
        """
        # Initialize images
        range_semantic = np.zeros((height, width, 3), dtype=np.uint8)
        range_instance = np.zeros((height, width, 3), dtype=np.uint8)
        range_depth = np.zeros((height, width), dtype=np.float32)
        
        # Convert to spherical coordinates
        depth = np.linalg.norm(points, axis=1)
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        
        # Calculate angles
        yaw = np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / (depth + 1e-8))
        
        # Convert to image coordinates
        fov_up = fov_up * np.pi / 180
        fov_down = fov_down * np.pi / 180
        fov = abs(fov_down) + abs(fov_up)
        
        proj_x = 0.5 * (yaw / np.pi + 1.0) * width
        proj_y = (1.0 - (pitch + abs(fov_down)) / fov) * height
        
        # Round to nearest pixel
        proj_x = np.floor(proj_x).astype(np.int32)
        proj_y = np.floor(proj_y).astype(np.int32)
        
        # Filter valid projections
        valid = (proj_x >= 0) & (proj_x < width) & (proj_y >= 0) & (proj_y < height)
        proj_x = proj_x[valid]
        proj_y = proj_y[valid]
        depth_valid = depth[valid]
        
        if semantic_labels is not None:
            semantic_valid = semantic_labels[valid]
        if instance_labels is not None:
            instance_valid = instance_labels[valid]
        
        # Fill images (keep closest point at each pixel)
        for i in range(len(proj_x)):
            if range_depth[proj_y[i], proj_x[i]] == 0 or depth_valid[i] < range_depth[proj_y[i], proj_x[i]]:
                range_depth[proj_y[i], proj_x[i]] = depth_valid[i]
                
                if semantic_labels is not None:
                    color = self.semantic_colors.get(semantic_valid[i], [128, 128, 128])
                    range_semantic[proj_y[i], proj_x[i]] = color[::-1]  # BGR
                
                if instance_labels is not None:
                    color = self.instance_colors.get(instance_valid[i], [128, 128, 128])
                    range_instance[proj_y[i], proj_x[i]] = color[::-1]  # BGR
        
        # Normalize depth for visualization
        range_depth_vis = (range_depth / range_depth.max() * 255).astype(np.uint8)
        range_depth_vis = cv2.applyColorMap(range_depth_vis, cv2.COLORMAP_JET)
        
        return range_semantic, range_instance, range_depth_vis
    
    def create_legend(self, detected_objects, max_items=20):
        """Create a legend showing detected objects"""
        # Create legend image
        legend_height = min(len(detected_objects) + 1, max_items) * 25 + 50
        legend = np.ones((legend_height, 300, 3), dtype=np.uint8) * 255
        
        # Title
        cv2.putText(legend, "Detected Objects", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add each object
        y_pos = 60
        for i, obj in enumerate(detected_objects[:max_items]):
            if y_pos > legend_height - 20:
                break
            
            # Draw color box
            color = self.instance_colors.get(obj['instance_id'], [128, 128, 128])
            cv2.rectangle(legend, (10, y_pos - 15), (30, y_pos), color[::-1], -1)
            
            # Add text
            text = f"ID:{obj['instance_id']} Class:{obj['class_id']} ({obj['num_points']} pts)"
            cv2.putText(legend, text, (40, y_pos - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            y_pos += 25
        
        return legend
    
    def visualize(self, results_dir, filename_base, show_range=True, save_output=False):
        """
        Main visualization function
        
        Args:
            results_dir: Directory containing results
            filename_base: Base filename of results
            show_range: Whether to show range image view
            save_output: Whether to save visualization to file
        """
        print(f"Loading results from {results_dir}/{filename_base}...")
        
        # Load results
        points, intensity, semantic_labels, instance_labels, detected_objects = \
            self.load_results(results_dir, filename_base)
        
        print(f"Loaded {len(points)} points with {len(detected_objects)} detected objects")
        
        # Create BEV projections
        print("Creating Bird's Eye View projections...")
        bev_semantic, bev_instance = self.project_to_bev(
            points, semantic_labels, instance_labels,
            x_range=(-40, 40), y_range=(-40, 40), resolution=0.1
        )
        
        # Resize for better visibility
        scale = 2
        bev_semantic = cv2.resize(bev_semantic, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        bev_instance = cv2.resize(bev_instance, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        # Add labels
        cv2.putText(bev_semantic, "Semantic Segmentation (BEV)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bev_instance, "Instance Segmentation (BEV)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Create range images if requested
        if show_range:
            print("Creating range image projections...")
            range_semantic, range_instance, range_depth = self.project_to_range_image(
                points, semantic_labels, instance_labels
            )
            
            # Scale up for visibility
            range_scale = 4
            range_semantic = cv2.resize(range_semantic, None, fx=1, fy=range_scale, interpolation=cv2.INTER_NEAREST)
            range_instance = cv2.resize(range_instance, None, fx=1, fy=range_scale, interpolation=cv2.INTER_NEAREST)
            range_depth = cv2.resize(range_depth, None, fx=1, fy=range_scale, interpolation=cv2.INTER_NEAREST)
            
            # Add labels
            cv2.putText(range_semantic, "Semantic (Range)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(range_instance, "Instance (Range)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(range_depth, "Depth (Range)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create legend
        legend = self.create_legend(detected_objects)
        
        # Display windows
        print("Displaying visualizations... Press 'q' to quit, 's' to save")
        
        cv2.namedWindow("Semantic BEV", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Instance BEV", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Legend", cv2.WINDOW_NORMAL)
        
        cv2.imshow("Semantic BEV", bev_semantic)
        cv2.imshow("Instance BEV", bev_instance)
        cv2.imshow("Legend", legend)
        
        if show_range:
            cv2.namedWindow("Range Views", cv2.WINDOW_NORMAL)
            range_combined = np.vstack([range_semantic, range_instance, range_depth])
            cv2.imshow("Range Views", range_combined)
        
        # Wait for key press
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_output = True
                break
        
        # Save output if requested
        if save_output:
            output_dir = Path(results_dir) / "visualizations"
            output_dir.mkdir(exist_ok=True)
            
            cv2.imwrite(str(output_dir / f"{filename_base}_bev_semantic.png"), bev_semantic)
            cv2.imwrite(str(output_dir / f"{filename_base}_bev_instance.png"), bev_instance)
            cv2.imwrite(str(output_dir / f"{filename_base}_legend.png"), legend)
            
            if show_range:
                cv2.imwrite(str(output_dir / f"{filename_base}_range_combined.png"), range_combined)
            
            print(f"Saved visualizations to {output_dir}")
        
        cv2.destroyAllWindows()


@click.command()
@click.argument('results_dir', type=click.Path(exists=True))
@click.argument('filename_base')
@click.option('--no_range', is_flag=True, help='Skip range image visualization')
@click.option('--save', is_flag=True, help='Save visualizations to file')
def main(results_dir, filename_base, no_range, save):
    """
    Visualize point cloud segmentation results
    
    Examples:
        python visualize_results.py results/ pointcloud_000000
        python visualize_results.py results/ pointcloud_000000 --save
        python visualize_results.py results/ pointcloud_000000 --no_range
    """
    visualizer = PointCloudVisualizer()
    visualizer.visualize(results_dir, filename_base, 
                        show_range=not no_range, save_output=save)


if __name__ == "__main__":
    main()