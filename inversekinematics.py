import numpy as np
from dh import Joint, ForwardKinematicsDH


class InverseKinematics:
    def __init__(self, fk, max_iterations=1000, tolerance=1e-6):
        self.fk = fk
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def compute_ik(self, desired_position):
        current_joint_angles = np.array([joint.theta for joint in self.fk.joints])
        
        for _ in range(self.max_iterations):
            current_end_effector = self.fk.compute_end_effector()
            current_position = current_end_effector[:3, 3]
            
            error = desired_position - current_position
            
            if np.linalg.norm(error) < self.tolerance:
                break
            
            J = self.fk.compute_jacobian()
            
            delta_theta = np.dot(np.linalg.pinv(J[:3, :]), error)
            current_joint_angles += delta_theta
            
            for i, joint in enumerate(self.fk.joints):
                joint.theta = current_joint_angles[i]
        
        return current_joint_angles

