"""Local checks only; these do NOT substitute for running the CARLA tests.

Run with: python -m unittest -v test_controller
"""
import math
from types import SimpleNamespace
import unittest

from carla_collision_avoidance import (
    Readings, lateral_target, speed_command, steering_command,
)


class ControllerTests(unittest.TestCase):
    def test_sudden_close_obstacle_cuts_throttle_and_commands_full_brake(self):
        for speed in (0.0, 1.0, 3.0, 5.0):
            with self.subTest(speed=speed):
                throttle, brake = speed_command(speed, 5.0, 1.0)
                self.assertEqual(throttle, 0.0)
                self.assertEqual(brake, 1.0)

    def test_stops_with_standoff_in_simple_longitudinal_model(self):
        # Simplified dynamics exercise the actual control helper over time.
        # They are deliberately not a claim about Unreal/vehicle physics.
        for initial_speed in (0.0, 3.0, 5.0):
            gap, speed = 40.0, initial_speed
            for _ in range(1000):
                throttle, brake = speed_command(speed, 5.0, gap)
                self.assertFalse(throttle > 0 and brake > 0)
                speed = max(0, speed + (3 * throttle - 3 * brake) * 0.02)
                gap -= speed * 0.02
            self.assertLess(speed, 0.15)
            self.assertGreater(gap, 1.0)

    def test_avoidance_path_clears_parked_car_in_bicycle_model(self):
        for offset in (-3.5, 3.5):
            x = y = yaw = 0.0
            speed, wheelbase, limit = 5.0, 2.9, math.radians(70)
            for _ in range(620):
                target_x = x + 4.0
                target_y = lateral_target(target_x, 2.0, offset)
                dx, dy = target_x - x, target_y - y
                local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
                local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
                steer = steering_command(local_x, local_y, wheelbase, limit)
                yaw += speed / wheelbase * math.tan(steer * limit) * 0.02
                x += speed * math.cos(yaw) * 0.02
                y += speed * math.sin(yaw) * 0.02
                if abs(x - 45) < 4.8:
                    self.assertGreater(abs(y), 2.0, "Car body overlaps parked car")
            self.assertGreater(x, 53)
            self.assertLess(abs(y - offset), 0.75)

    def test_sensor_retains_nearest_fresh_event_and_expires_old_events(self):
        readings = Readings()
        for actor_id, stamp, distance in ((1, 2.0, 8), (2, 2.05, 30)):
            readings.obstacle(SimpleNamespace(
                other_actor=SimpleNamespace(id=actor_id), timestamp=stamp, distance=distance))
        gap, seen, collisions = readings.snapshot(2.1)
        self.assertEqual(gap, 8)
        self.assertEqual(seen, {1, 2})
        self.assertEqual(collisions, [])
        self.assertTrue(math.isinf(readings.snapshot(2.5)[0]))

    def test_collision_callback_records_the_hit_actor(self):
        readings = Readings()
        readings.collision(SimpleNamespace(
            frame=123, other_actor=SimpleNamespace(id=42, type_id="vehicle.tesla.model3")))
        _, _, collisions = readings.snapshot(10.0)
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0]["actor_id"], 42)
        self.assertEqual(collisions[0]["frame"], 123)
        # Reading or clearing a returned list must not lose the recorded hit.
        collisions.clear()
        self.assertEqual(len(readings.snapshot(10.0)[2]), 1)


if __name__ == "__main__":
    unittest.main()
