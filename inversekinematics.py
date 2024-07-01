import numpy as np
from dh import Joint, ForwardKinematicsDH


class InverseKinematics:
    def __init__(self, fk, max_iterations=100, tolerance=1e-6, error_count=0):
        self.fk = fk
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.error_count = error_count
    
    def error_Counter(self):
        return print("A quantidade de tentativas foram: ",self.error_count)

    def compute_ik(self, desired_position):
        current_joint_angles = np.array([joint.theta for joint in self.fk.joints])
        
        for _ in range(self.max_iterations):
            current_end_effector = self.fk.compute_end_effector()
            current_position = current_end_effector[:3, 3]
            
            error = desired_position - current_position
            
            if np.linalg.norm(error) < self.tolerance:
                break
            else:
                self.error_count += 1
                if self.error_count > self.max_iterations:
                    raise Exception("Não foi possível encontrar uma solução.")
            
            J = self.fk.compute_jacobian()
            
            delta_theta = np.dot(np.linalg.pinv(J[:3, :]), error)
            current_joint_angles += delta_theta
            
            for i, joint in enumerate(self.fk.joints):
                joint.theta = current_joint_angles[i]
        
        return current_joint_angles

