#!/usr/bin/env python3
"""
Windows PLY Viewer - Visualize segmentation results
Install: pip install open3d numpy
"""

import numpy as np
import argparse
from pathlib import Path
import struct


class PLYViewer:
    """Interactive PLY viewer for Windows"""
    
    def __init__(self):
        try:
            import open3d as o3d
            self.o3d = o3d
        except ImportError:
            print("Please install Open3D:")
            print("  pip install open3d")
            exit(1)
        
        self.class_names = ['unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 
                           'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
                           'road', 'parking', 'sidewalk', 'other-ground', 'building',
                           'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
    
    def load_ply(self, ply_path):
        """Load PLY file with custom properties"""
        print(f"Loading {ply_path}")
        
        # Use Open3D to load the basic point cloud
        pcd = self.o3d.io.read_point_cloud(str(ply_path))
        
        # Also parse to get semantic/instance data
        semantic, instance = self.parse_ply_properties(ply_path)
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        print(f"  Loaded {len(points)} points")
        
        if semantic is not None:
            unique_classes, counts = np.unique(semantic, return_counts=True)
            print(f"  Found {len(unique_classes)} semantic classes")
        
        if instance is not None:
            num_instances = len(np.unique(instance)) - 1
            print(f"  Found {num_instances} instances")
        
        return pcd, points, colors, semantic, instance
    
    def parse_ply_properties(self, ply_path):
        """Parse PLY file to extract semantic and instance properties"""
        try:
            with open(ply_path, 'rb') as f:
                # Read header
                header = b''
                while True:
                    line = f.readline()
                    header += line
                    if b'end_header' in line:
                        break
                
                # Parse header to find properties
                header_str = header.decode('ascii')
                lines = header_str.split('\n')
                
                num_vertices = 0
                has_semantic = False
                has_instance = False
                
                for line in lines:
                    if 'element vertex' in line:
                        num_vertices = int(line.split()[-1])
                    elif 'property' in line and 'semantic' in line:
                        has_semantic = True
                    elif 'property' in line and 'instance' in line:
                        has_instance = True
                
                if not has_semantic and not has_instance:
                    return None, None
                
                # Read binary data
                semantic = []
                instance = []
                
                for i in range(num_vertices):
                    # Read x, y, z (3 floats)
                    f.read(12)
                    # Read r, g, b (3 bytes)
                    f.read(3)
                    
                    if has_semantic:
                        # Read semantic (int32)
                        sem = struct.unpack('<i', f.read(4))[0]
                        semantic.append(sem)
                    
                    if has_instance:
                        # Read instance (int32)
                        ins = struct.unpack('<i', f.read(4))[0]
                        instance.append(ins)
                
                semantic = np.array(semantic) if semantic else None
                instance = np.array(instance) if instance else None
                
                return semantic, instance
                
        except Exception as e:
            print(f"  Could not parse extended properties: {e}")
            return None, None
    
    def visualize(self, ply_path, color_mode='original', save_screenshot=None):
        """
        Visualize PLY file
        
        Args:
            ply_path: Path to PLY file
            color_mode: 'original', 'semantic', 'instance', 'height'
            save_screenshot: Path to save screenshot
        """
        # Load PLY
        pcd, points, colors, semantic, instance = self.load_ply(ply_path)
        
        # Apply coloring based on mode
        if color_mode == 'semantic' and semantic is not None:
            print("  Applying semantic coloring")
            colors = self.apply_semantic_colors(semantic)
            pcd.colors = self.o3d.utility.Vector3dVector(colors)
            
        elif color_mode == 'instance' and instance is not None:
            print("  Applying instance coloring")
            colors = self.apply_instance_colors(instance)
            pcd.colors = self.o3d.utility.Vector3dVector(colors)
            
        elif color_mode == 'height':
            print("  Applying height coloring")
            colors = self.apply_height_colors(points)
            pcd.colors = self.o3d.utility.Vector3dVector(colors)
        
        # Create visualization window
        vis = self.o3d.visualization.Visualizer()
        vis.create_window(window_name=f"PLY Viewer - {Path(ply_path).name}", 
                         width=1280, height=720)
        
        # Add geometry
        vis.add_geometry(pcd)
        
        # Add coordinate frame
        coord_frame = self.o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        vis.add_geometry(coord_frame)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
        
        # Set initial viewpoint
        view_control = vis.get_view_control()
        view_control.set_zoom(0.5)
        view_control.set_front([0.5, -0.5, -0.3])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 0, 1])
        
        if save_screenshot:
            # Update and capture
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(save_screenshot))
            print(f"  Screenshot saved to {save_screenshot}")
            vis.destroy_window()
        else:
            # Interactive mode
            print("\nControls:")
            print("  Mouse: Rotate | Scroll: Zoom | Ctrl+Mouse: Pan")
            print("  +/-: Increase/decrease point size")
            print("  R: Reset view | Q: Quit")
            
            vis.run()
            vis.destroy_window()
        
        # Print statistics if available
        if semantic is not None:
            self.print_statistics(semantic, instance)
    
    def apply_semantic_colors(self, semantic):
        """Apply semantic class colors"""
        color_map = np.array([
            [0.0, 0.0, 0.0],      # 0: unlabeled - black
            [0.96, 0.59, 0.39],   # 1: car - orange
            [0.96, 0.90, 0.39],   # 2: bicycle - yellow
            [0.59, 0.24, 0.12],   # 3: motorcycle - brown
            [0.71, 0.12, 0.31],   # 4: truck - purple
            [1.00, 0.00, 0.00],   # 5: other-vehicle - red
            [0.12, 0.12, 1.00],   # 6: person - blue
            [0.78, 0.16, 1.00],   # 7: bicyclist - violet
            [0.35, 0.12, 0.59],   # 8: motorcyclist - dark purple
            [1.00, 0.00, 1.00],   # 9: road - magenta
            [1.00, 0.59, 1.00],   # 10: parking - light magenta
            [0.29, 0.00, 0.29],   # 11: sidewalk - dark purple
            [0.29, 0.00, 0.69],   # 12: other-ground - purple
            [0.00, 0.78, 1.00],   # 13: building - cyan
            [0.20, 0.47, 1.00],   # 14: fence - light blue
            [0.00, 0.69, 0.00],   # 15: vegetation - green
            [0.00, 0.24, 0.53],   # 16: trunk - dark blue
            [0.31, 0.94, 0.59],   # 17: terrain - light green
            [0.59, 0.94, 1.00],   # 18: pole - light cyan
            [0.00, 0.00, 1.00],   # 19: traffic-sign - blue
        ])
        
        # Extend color map if needed
        if semantic.max() >= len(color_map):
            extra_colors = np.random.rand(semantic.max() - len(color_map) + 1, 3)
            color_map = np.vstack([color_map, extra_colors])
        
        return color_map[semantic]
    
    def apply_instance_colors(self, instance):
        """Apply random colors for instances"""
        num_instances = instance.max() + 1
        
        # Generate distinct colors
        np.random.seed(42)  # For consistency
        instance_colors = np.random.rand(num_instances, 3)
        instance_colors[0] = [0.5, 0.5, 0.5]  # Gray for no instance
        
        # Make colors more vibrant
        instance_colors = instance_colors * 0.7 + 0.3
        
        return instance_colors[instance]
    
    def apply_height_colors(self, points):
        """Apply gradient coloring based on height"""
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        
        # Colormap: blue (low) -> green -> yellow -> red (high)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = np.clip(2 * z_norm - 1, 0, 1)  # Red
        colors[:, 1] = np.clip(2 - 2 * abs(z_norm - 0.5), 0, 1)  # Green
        colors[:, 2] = np.clip(1 - 2 * z_norm, 0, 1)  # Blue
        
        return colors
    
    def print_statistics(self, semantic, instance):
        """Print segmentation statistics"""
        print("\n" + "="*50)
        print("Segmentation Statistics:")
        print("="*50)
        
        # Semantic statistics
        unique_classes, counts = np.unique(semantic, return_counts=True)
        total_points = len(semantic)
        
        print(f"\nSemantic Classes ({len(unique_classes)} total):")
        for cls_id, count in zip(unique_classes, counts):
            percentage = count / total_points * 100
            if cls_id < len(self.class_names):
                name = self.class_names[cls_id]
            else:
                name = f"Class_{cls_id}"
            print(f"  {name:20} : {count:7} points ({percentage:5.1f}%)")
        
        # Instance statistics
        if instance is not None:
            unique_instances = np.unique(instance)
            num_instances = len(unique_instances) - 1  # Exclude 0
            
            print(f"\nInstances: {num_instances} detected")
            
            if num_instances > 0:
                instance_sizes = []
                for inst_id in unique_instances:
                    if inst_id > 0:
                        size = (instance == inst_id).sum()
                        instance_sizes.append(size)
                
                if instance_sizes:
                    print(f"  Min size: {min(instance_sizes)} points")
                    print(f"  Max size: {max(instance_sizes)} points")
                    print(f"  Avg size: {np.mean(instance_sizes):.0f} points")
    
    def batch_visualize(self, ply_dir, output_dir=None):
        """Create screenshots for all PLY files in directory"""
        ply_files = list(Path(ply_dir).glob("*.ply"))
        print(f"Found {len(ply_files)} PLY files")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for ply_file in ply_files:
            print(f"\nProcessing {ply_file.name}")
            
            if output_dir:
                screenshot_path = output_dir / f"{ply_file.stem}.png"
                self.visualize(str(ply_file), save_screenshot=screenshot_path)
            else:
                self.visualize(str(ply_file))


def main():
    parser = argparse.ArgumentParser(description='PLY file viewer for Windows')
    parser.add_argument('input', help='PLY file or directory')
    parser.add_argument('--color', choices=['original', 'semantic', 'instance', 'height'],
                       default='original', help='Coloring mode')
    parser.add_argument('--screenshot', help='Save screenshot to file')
    parser.add_argument('--batch-screenshots', help='Output directory for batch screenshots')
    
    args = parser.parse_args()
    
    viewer = PLYViewer()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        viewer.visualize(str(input_path), color_mode=args.color, 
                        save_screenshot=args.screenshot)
    else:
        # Directory
        viewer.batch_visualize(str(input_path), args.batch_screenshots)


if __name__ == "__main__":
    main()