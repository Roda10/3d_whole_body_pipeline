import numpy as np
import torch

class TemporalStabilizer:
    def __init__(self, window_size=5, smoothing_factor=0.8, mode='single'):
        """
        Initialize the temporal stabilizer
        Args:
            window_size (int): Size of temporal window for smoothing
            smoothing_factor (float): Factor for exponential smoothing (0-1)
            mode (str): Operating mode - 'single' or 'video'
        """
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.mode = mode
        self.pose_history = []
        self.velocity_history = []
        
        # Parameters for pose validation
        self.angle_limits = {
            'body_pose': (-np.pi, np.pi),  # Global limits for body joints
            'left_hand_pose': (-np.pi/2, np.pi/2),  # More restrictive for fingers
            'right_hand_pose': (-np.pi/2, np.pi/2),
            'jaw_pose': (-np.pi/4, np.pi/4)  # Limited range for jaw
        }
        
    def validate_pose(self, pose_params):
        """
        Validate pose parameters against anatomical constraints
        Args:
            pose_params: Dictionary of pose parameters
        Returns:
            validated_params: Dictionary with constrained parameters
        """
        validated = pose_params.copy()
        
        for param_name, (min_angle, max_angle) in self.angle_limits.items():
            if param_name in validated and isinstance(validated[param_name], np.ndarray):
                validated[param_name] = np.clip(validated[param_name], min_angle, max_angle)
        
        return validated

    def stabilize_pose(self, current_pose, is_keyframe=False):
        """
        Stabilize pose parameters using temporal smoothing and motion constraints
        Args:
            current_pose: Dictionary containing pose parameters
            is_keyframe: Boolean indicating if this is a keyframe (for video mode)
        Returns:
            Stabilized pose parameters
        """
        # Convert pose parameters to numpy if they're torch tensors
        pose_params = {}
        for key, value in current_pose.items():
            if isinstance(value, torch.Tensor):
                pose_params[key] = value.detach().cpu().numpy()
            else:
                pose_params[key] = value
                
        # Validate pose against anatomical constraints
        pose_params = self.validate_pose(pose_params)

        # Add to history
        self.pose_history.append(pose_params)
        if len(self.pose_history) > self.window_size:
            self.pose_history.pop(0)

        # Not enough frames for smoothing yet
        if len(self.pose_history) < 2:
            return current_pose

        # Calculate velocities
        velocity = {}
        for key in pose_params.keys():
            if isinstance(self.pose_history[-1][key], np.ndarray):
                velocity[key] = self.pose_history[-1][key] - self.pose_history[-2][key]

        self.velocity_history.append(velocity)
        if len(self.velocity_history) > self.window_size:
            self.velocity_history.pop(0)

        # Apply temporal smoothing
        smoothed_pose = {}
        for key in pose_params.keys():
            if isinstance(pose_params[key], np.ndarray):
                # Exponential smoothing
                history = np.stack([frame[key] for frame in self.pose_history])
                weights = np.power(self.smoothing_factor, np.arange(len(history)-1, -1, -1))
                weights = weights / weights.sum()
                smoothed_pose[key] = np.sum(history * weights[:, None], axis=0)

                # Velocity-based stabilization
                if len(self.velocity_history) > 1:
                    mean_velocity = np.mean([v[key] for v in self.velocity_history[:-1]], axis=0)
                    current_velocity = self.velocity_history[-1][key]
                    velocity_diff = current_velocity - mean_velocity
                    
                    # Dampen sudden changes
                    damping_factor = 0.5
                    smoothed_pose[key] = smoothed_pose[key] - velocity_diff * damping_factor
            else:
                smoothed_pose[key] = pose_params[key]

        # Convert back to torch tensors if needed
        stabilized_pose = {}
        for key, value in smoothed_pose.items():
            if isinstance(current_pose[key], torch.Tensor):
                stabilized_pose[key] = torch.from_numpy(value).to(current_pose[key].device)
            else:
                stabilized_pose[key] = value

        return stabilized_pose

    def reset(self):
        """Reset the stabilizer's history"""
        self.pose_history = []
        self.velocity_history = []
        
    def set_mode(self, mode):
        """
        Switch between single-image and video modes
        Args:
            mode (str): 'single' or 'video'
        """
        if mode not in ['single', 'video']:
            raise ValueError("Mode must be either 'single' or 'video'")
        self.mode = mode
        self.reset()  # Reset history when changing modes
        
    def update_angle_limits(self, new_limits):
        """
        Update anatomical angle limits
        Args:
            new_limits (dict): Dictionary of joint angle limits
        """
        self.angle_limits.update(new_limits)