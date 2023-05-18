
import numpy as np
import pygame
from rlib.envs.walker2d.utils import get_length, get_angle


class Bone:

    def __init__(self, pos_start, pos_end, weight, thickness=0.005, color=(255, 255, 255)):

        self.pos_start = np.array(pos_start)
        self.pos_end = np.array(pos_end)
        self.weight = weight
        self.thickness = thickness
        self.color = color

        self.length = np.linalg.norm(self.pos_end - self.pos_start)
        self.moment_of_inertia = self.weight * self.length ** 2 / 12

        self.angular_force = 0
        self.cartesian_force = np.array([0, 0]).astype(float)

        self.angular_velocity = 0
        self.cartesian_velocity = np.array([0, 0]).astype(float)

        self.set_rotation_point(0.5)

    def set_rotation_point(self, rotation_pos):
        self.rotation_pos = rotation_pos
        self.rotation_point = self.pos_start + self.rotation_pos * (self.pos_end - self.pos_start)
        self.angle = get_angle(self.pos_end - self.rotation_point, np.array([1, 0]))
        if np.dot(self.pos_end - self.rotation_point, np.array([0, 1])) < 0:
            self.angle = 2 * np.pi - self.angle

    def apply_force(self, angular_force=0, cartesian_force=np.array([0, 0]).astype(float), position_force=0):

        assert 0 <= position_force <= 1, "The position must be between 0 and 1"

        self.angular_force += angular_force

        if not np.allclose(cartesian_force, np.array([0, 0])):

            # Add the linear force
            self.cartesian_force += cartesian_force

            # Compute its torque
            lever_arm_vector = self.pos_start + position_force * (self.pos_end - self.pos_start) - self.rotation_point
            torque = np.cross(lever_arm_vector, cartesian_force)

            # Add the torque to the angular force
            self.angular_force += torque

    def update(self, dt=0.01):

        # First apply the linear force
        acceleration = self.cartesian_force / self.weight
        self.cartesian_velocity += acceleration * dt
        self.pos_start += self.cartesian_velocity * dt
        self.pos_end += self.cartesian_velocity * dt

        self.set_rotation_point(self.rotation_pos)

        # Now apply the angular force
        acceleration = self.angular_force / self.moment_of_inertia
        self.angular_velocity += acceleration * dt
        self.angle += self.angular_velocity * dt

        # Update the position of the start
        self.pos_start = self.rotation_point + np.array([np.cos(self.angle), np.sin(self.angle)]) * np.linalg.norm(self.rotation_point - self.pos_start)

        # Update the position of the end
        self.pos_end = self.rotation_point - np.array([np.cos(self.angle), np.sin(self.angle)]) * np.linalg.norm(self.rotation_point - self.pos_end)

        # Reset the forces
        self.angular_force = 0
        self.cartesian_force = np.array([0, 0]).astype(float)

        self.set_rotation_point(self.rotation_pos)

    def __repr__(self) -> str:
        string = "Bone:\n"
        string += "Start: {}\n".format(self.pos_start)
        string += "End: {}\n".format(self.pos_end)
        string += "Rotation Point: {}\n".format(self.center)
        string += "Weight: {}\n".format(self.weight)
        string += "Length: {}\n".format(self.length)
        string += "Moment of inertia: {}\n".format(self.moment_of_inertia)
        string += "Angular force: {}\n".format(self.angular_force)
        string += "Cartesian force: {}\n".format(self.cartesian_force)
        string += "Angular velocity: {}\n".format(self.angular_velocity)
        string += "Cartesian velocity: {}\n".format(self.cartesian_velocity)
        string += "Angle: {}\n".format(self.angle)
        string += "Thickness: {}\n".format(self.thickness)
        string += "Color: {}\n".format(self.color)
        return string


    def draw(self, screen):
        
        screen_size = np.array(screen.get_size())
        perp_vector = self.pos_end - self.pos_start
        perp_vector = np.array([-perp_vector[1], perp_vector[0]])
        perp_vector /= np.linalg.norm(perp_vector)

        point1 = self.pos_start + self.thickness * perp_vector / 2
        point2 = self.pos_start - self.thickness * perp_vector / 2
        point3 = self.pos_end - self.thickness * perp_vector / 2
        point4 = self.pos_end + self.thickness * perp_vector / 2

        point1 = (point1 * screen_size).astype(int)
        point2 = (point2 * screen_size).astype(int)
        point3 = (point3 * screen_size).astype(int)
        point4 = (point4 * screen_size).astype(int)

        pygame.draw.polygon(screen, self.color, [np.array(point1), point2, point3, point4])

    


class BONE1:

    def __init__(self, start, end, weight, thickness, color=(255, 255, 255)):
        self.start = np.array(start)
        self.end = np.array(end)
        self.length = get_length(self.end - self.start)
        self.weight = weight
        self.thickness = thickness
        self.color = color
        self.angular_force_start = 0
        self.angular_force_end = 0
        self.cartesian_force_start = np.array([0, 0]).astype(float)
        self.cartesian_force_end = np.array([0, 0]).astype(float)

        self.angular_velocity_start = 0
        self.angular_velocity_end = 0
        self.cartesian_velocity_start = np.array([0, 0]).astype(float)
        self.cartesian_velocity_end = np.array([0, 0]).astype(float)

        self.angle_start = get_angle(self.start - self.end, np.array([1, 0]))
        if np.dot(self.start - self.end, np.array([0, 1])) < 0:
            self.angle_start = 2 * np.pi - self.angle_start

        self.angle_end = get_angle(self.end - self.start, np.array([1, 0]))
        if np.dot(self.end - self.start, np.array([0, 1])) < 0:
            self.angle_end = 2 * np.pi - self.angle_end

    def apply_force(self,
                    angular_force_start=0,
                    angular_force_end=0,
                    cartesian_force_start=np.array([0, 0]).astype(float), 
                    cartesian_force_end=np.array([0, 0]).astype(float)):
        
        self.angular_force_start += angular_force_start
        self.angular_force_end += angular_force_end
        self.cartesian_force_start += cartesian_force_start
        self.cartesian_force_end += cartesian_force_end

        # First, determine the final angular force
        if np.sign(self.angular_force_start) == np.sign(self.angular_force_end):
            self.angular_force_start += self.angular_force_end
            self.angular_force_end = 0
        else:
            if np.abs(self.angular_force_start) > np.abs(self.angular_force_end):
                self.angular_force_start += self.angular_force_end
                self.angular_force_end = 0
            else:
                self.angular_force_end += self.angular_force_start
                self.angular_force_start = 0

    def __repr__(self) -> str:
        print("Bone:")
        print("Start: {}".format(self.start))
        print("End: {}".format(self.end))
        print("Length: {}".format(self.length))
        print("Weight: {}".format(self.weight))
        print("Thickness: {}".format(self.thickness))
        print("Color: {}".format(self.color))
        print("Angular force start: {}".format(self.angular_force_start))
        print("Angular force end: {}".format(self.angular_force_end))
        print("Cartesian force start: {}".format(self.cartesian_force_start))
        print("Cartesian force end: {}".format(self.cartesian_force_end))
        print("Angular velocity start: {}".format(self.angular_velocity_start))
        print("Angular velocity end: {}".format(self.angular_velocity_end))
        print("Cartesian velocity start: {}".format(self.cartesian_velocity_start))
        print("Cartesian velocity end: {}".format(self.cartesian_velocity_end))
        return ""

    
    def update(self, dt=0.01):
        
        old_start = self.start.copy()

        # Get the current torques and angles
        torque_start = self.angular_force_start * self.length
        torque_end = self.angular_force_end * self.length

        self.angle_start = get_angle(self.start - self.end, np.array([1, 0]))
        if np.dot(self.start - self.end, np.array([0, 1])) < 0:
            self.angle_start = 2 * np.pi - self.angle_start

        self.angle_end = get_angle(self.end - self.start, np.array([1, 0]))
        if np.dot(self.end - self.start, np.array([0, 1])) < 0:
            self.angle_end = 2 * np.pi - self.angle_end
        
        # First do it on the start end of the bone
        # Using the angular force
        # Compute the acceleration
        acceleration = torque_start / self.weight

        # Compute the velocity
        self.angular_velocity_start += acceleration * dt

        # Compute the new position
        self.angle_start += self.angular_velocity_start * dt
        self.start = self.end + np.array([np.cos(self.angle_start), np.sin(self.angle_start)]) * self.length

        # Apply the cartesian force
        acceleration = self.cartesian_force_start / self.weight

        # Compute the velocity
        self.cartesian_velocity_start += acceleration * dt

        # Compute the new position
        self.start += self.cartesian_velocity_start * dt

        # Now do it on the end end of the bone
        # Compute the acceleration
        acceleration = torque_end / self.weight

        # Compute the velocity
        self.angular_velocity_end += acceleration * dt

        # Compute the new position
        self.angle_end += self.angular_velocity_end * dt
        self.end = old_start + np.array([np.cos(self.angle_end), np.sin(self.angle_end)]) * self.length

        # Using the cartesian force
        # Compute the acceleration
        acceleration = self.cartesian_force_end / self.weight

        # Compute the velocity
        self.cartesian_velocity_end += acceleration * dt

        # Compute the new position
        self.end += self.cartesian_velocity_end * dt

        self.angular_force_start = 0
        self.angular_force_end = 0
        self.cartesian_force_start = np.array([0, 0]).astype(float)
        self.cartesian_force_end = np.array([0, 0]).astype(float)


    def set_velocity(self, angular_start, angular_end, cartesian_start, cartesian_end):
        self.angular_velocity_start = angular_start
        self.angular_velocity_end = angular_end
        self.cartesian_velocity_start = cartesian_start
        self.cartesian_velocity_end = cartesian_end


    
    def draw(self, screen):
        
        screen_size = np.array(screen.get_size())
        perp_vector = self.end - self.start
        perp_vector = np.array([-perp_vector[1], perp_vector[0]])
        perp_vector /= np.linalg.norm(perp_vector)

        point1 = self.start + self.thickness * perp_vector / 2
        point2 = self.start - self.thickness * perp_vector / 2
        point3 = self.end - self.thickness * perp_vector / 2
        point4 = self.end + self.thickness * perp_vector / 2

        point1 = (point1 * screen_size).astype(int)
        point2 = (point2 * screen_size).astype(int)
        point3 = (point3 * screen_size).astype(int)
        point4 = (point4 * screen_size).astype(int)

        pygame.draw.polygon(screen, self.color, [np.array(point1), point2, point3, point4])
        