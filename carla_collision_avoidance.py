"""CARLA 0.9.15 straight-road obstacle avoidance tests (Python 3.10).

Run on the Windows/Linux computer with CARLA:
    python carla_collision_avoidance.py --check
    python carla_collision_avoidance.py --test brake
    python carla_collision_avoidance.py --test avoid
    python carla_collision_avoidance.py --test collision

The front obstacle sensor drives braking and initiates avoidance. Map geometry
guides steering. Nearby actor positions are used for checking the target lane;
this is simulator ground truth, not camera-based perception. This is a controlled
straight-road demonstration, not a general city-driving controller.
"""

import argparse
import json
import math
from pathlib import Path
import threading
import time

STEP = 0.05
SPEED = 5.0  # m/s (18 km/h)


def clamp(value, low, high):
    return max(low, min(high, value))


def brake_distance(speed):
    # Conservative demo threshold: stand-off + reaction distance + braking.
    return 5.0 + speed * 0.7 + speed * speed / (2 * 3.0)


def lateral_target(progress, start, offset):
    if start is None:
        return 0.0
    fraction = clamp((progress - start) / 20.0, 0.0, 1.0)
    return offset * fraction * fraction * (3.0 - 2.0 * fraction)


def steering_command(forward_distance, right_distance, wheelbase, limit):
    curvature = 2.0 * right_distance / max(
        0.1, forward_distance**2 + right_distance**2)
    return clamp(math.atan(wheelbase * curvature) / max(0.1, limit), -0.6, 0.6)


def speed_command(speed, desired_speed, gap, avoidance_enabled=True):
    if avoidance_enabled and gap < brake_distance(speed):
        return 0.0, 1.0
    return (clamp(0.4 * (desired_speed - speed), 0, 0.6),
            clamp(0.3 * (speed - desired_speed), 0, 1))


class Readings:
    """Callbacks and the main loop share these values under a lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.recent = {}
        self.detected_ids = set()
        self.collisions = []

    def obstacle(self, event):
        with self.lock:
            self.recent[event.other_actor.id] = (event.timestamp, event.distance)
            self.detected_ids.add(event.other_actor.id)

    def collision(self, event):
        with self.lock:
            self.collisions.append({"frame": event.frame,
                                    "actor_id": event.other_actor.id,
                                    "type": event.other_actor.type_id})

    def snapshot(self, now):
        with self.lock:
            distances = [distance for stamp, distance in self.recent.values()
                         if 0 <= now - stamp <= 0.2]
            return (min(distances, default=math.inf),
                    set(self.detected_ids), list(self.collisions))


def dot_xy(a, b):
    return a.x * b.x + a.y * b.y


def straight_samples(waypoint):
    origin = waypoint.transform.location
    forward = waypoint.transform.get_forward_vector()
    right = waypoint.transform.get_right_vector()
    samples = [waypoint]
    for distance in range(5, 81, 5):
        choices = waypoint.next(float(distance))
        if len(choices) != 1:
            return None
        point = choices[0]
        delta = point.transform.location - origin
        if (point.is_junction or point.road_id != waypoint.road_id
                or point.lane_id != waypoint.lane_id
                or dot_xy(forward, point.transform.get_forward_vector()) < 0.995
                or abs(dot_xy(delta, right)) > 0.3
                or abs(delta.z) > 1.5):
            return None
        samples.append(point)
    return samples


def adjacent_lane(carla, samples):
    """Find a parallel, same-direction lane permitting the whole maneuver."""
    for direction, flag in (("left", carla.LaneChange.Left),
                            ("right", carla.LaneChange.Right)):
        neighbors = []
        for point in samples:
            other = (point.get_left_lane() if direction == "left"
                     else point.get_right_lane())
            if (other is None or other.lane_type != carla.LaneType.Driving
                    or other.is_junction
                    or dot_xy(point.transform.get_forward_vector(),
                              other.transform.get_forward_vector()) < 0.995
                    or (len(neighbors) < 7 and not (point.lane_change & flag))):
                break
            neighbors.append(other)
        if len(neighbors) == len(samples):
            return direction, neighbors
    return None, None


def create_scenario(carla, world, mode, actors):
    road_map = world.get_map()
    library = world.get_blueprint_library()
    blueprint = library.find("vehicle.tesla.model3")
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "avoidance_test")
    for spawn in road_map.get_spawn_points():
        waypoint = road_map.get_waypoint(spawn.location)
        if waypoint.is_junction:
            continue
        samples = straight_samples(waypoint)
        if not samples:
            continue
        direction, neighbors = adjacent_lane(carla, samples)
        if mode == "avoid" and neighbors is None:
            continue
        transform = waypoint.transform
        transform.location.z += 0.3
        ego = world.try_spawn_actor(blueprint, transform)
        if ego is None:
            continue
        actors.append(ego)
        obstacle_transform = samples[9].transform  # 45 m in front
        obstacle_transform.location.z += 0.3
        obstacle = world.try_spawn_actor(blueprint, obstacle_transform)
        if obstacle is None:
            ego.destroy()
            actors.remove(ego)
            continue
        actors.append(obstacle)
        obstacle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        offset = 0.0
        if neighbors:
            delta = neighbors[0].transform.location - waypoint.transform.location
            offset = dot_xy(delta, waypoint.transform.get_right_vector())
        return ego, obstacle, waypoint, offset, direction
    raise RuntimeError("No suitable empty 80 m straight road found. For avoidance, "
                       "a parallel lane with a broken lane marking is also needed. "
                       "Try Town04 or Town10HD in a fresh server session.")


def lane_clear(world, ego, origin, forward, right, progress, offset):
    # Deliberately uses simulator actor positions for this controlled test.
    for actor in world.get_actors():
        if actor.id == ego.id or not actor.type_id.startswith(("vehicle.", "walker.")):
            continue
        delta = actor.get_location() - origin
        longitudinal = dot_xy(delta, forward) - progress
        lateral = dot_xy(delta, right)
        if -20 < longitudinal < 65 and abs(lateral - offset) < 2.2:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--check", action="store_true", help="Check connection only")
    parser.add_argument("--test", choices=("brake", "avoid", "collision"), default="brake")
    parser.add_argument("--seconds", type=float, default=35.0)
    parser.add_argument("--output", type=Path, help="Optional JSON result filename")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    try:
        import carla
    except ImportError as error:
        raise SystemExit("CARLA's Python package is missing. On the Windows/Linux "
                         "computer with Python 3.10, run:\n"
                         "python -m pip install carla==0.9.15") from error

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    server_version = client.get_server_version()
    client_version = client.get_client_version()
    print(f"Connected: server {server_version}; client {client_version}")
    if server_version != client_version:
        raise RuntimeError("Use matching CARLA client and server versions.")
    if not server_version.startswith("0.9.15"):
        raise RuntimeError("This demonstration targets CARLA 0.9.15.")
    world = client.get_world()
    print("Map:", world.get_map().name)
    if args.check:
        print("CONNECTION OK. Ready to run tests.")
        return 0
    settings = world.get_settings()
    if settings.synchronous_mode:
        raise RuntimeError("The world is already synchronous. Stop the other tick "
                           "controller and start a fresh CARLA session for this test.")
    if any(a.type_id.startswith(("vehicle.", "walker.")) for a in world.get_actors()):
        raise RuntimeError("Use a fresh server with no spawned cars/pedestrians. "
                           "Stop traffic/manual_control scripts before testing.")

    actors = []
    sensors = []
    spectator = world.get_spectator()
    saved_view = spectator.get_transform()
    changed_settings = False
    readings = Readings()
    result = {"test": args.test, "passed": False, "reason": "Time limit reached"}
    try:
        temporary = world.get_settings()
        temporary.synchronous_mode = True
        temporary.fixed_delta_seconds = STEP
        world.apply_settings(temporary)
        changed_settings = True
        ego, obstacle, waypoint, offset, direction = create_scenario(
            carla, world, args.test, actors)
        origin = waypoint.transform.location
        forward = waypoint.transform.get_forward_vector()
        right = waypoint.transform.get_right_vector()
        library = world.get_blueprint_library()
        sensor_bp = library.find("sensor.other.obstacle")
        for name, value in {"distance": "50", "hit_radius": "0.8",
                            "only_dynamics": "false", "debug_linetrace": "true",
                            "sensor_tick": str(STEP)}.items():
            sensor_bp.set_attribute(name, value)
        sensor = world.spawn_actor(sensor_bp, carla.Transform(
            carla.Location(x=ego.bounding_box.extent.x + 0.3, z=1.2)), attach_to=ego)
        actors.append(sensor)
        sensors.append(sensor)
        sensor.listen(readings.obstacle)
        sensor = world.spawn_actor(library.find("sensor.other.collision"),
                                   carla.Transform(), attach_to=ego)
        actors.append(sensor)
        sensors.append(sensor)
        sensor.listen(readings.collision)

        ego.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(20):
            world.tick()
        shift_start = None
        stopped_for = 0.0
        top_speed = 0.0
        min_forward_gap = math.inf
        seen = set()
        collisions = []
        wheels = ego.get_physics_control().wheels
        steering_limit = math.radians(max(w.max_steer_angle for w in wheels))
        wheelbase = math.hypot(wheels[0].position.x - wheels[2].position.x,
                               wheels[0].position.y - wheels[2].position.y) / 100
        print(f"Starting {args.test} test. Watch the CARLA spectator window.")
        if args.test == "collision":
            print("This test deliberately drives into the parked car to verify the collision sensor.")
        for index in range(int(args.seconds / STEP)):
            wall_start = time.monotonic()
            snapshot = world.get_snapshot()
            now = snapshot.timestamp.elapsed_seconds
            gap, seen, collisions = readings.snapshot(now)
            transform = ego.get_transform()
            delta = transform.location - origin
            progress = dot_xy(delta, forward)
            lateral = dot_xy(delta, right)
            velocity = ego.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            top_speed = max(top_speed, speed)
            obstacle_delta = obstacle.get_location() - transform.location
            forward_gap = dot_xy(obstacle_delta, forward) - (
                ego.bounding_box.extent.x + obstacle.bounding_box.extent.x)
            if abs(lateral) < ego.bounding_box.extent.y + obstacle.bounding_box.extent.y:
                min_forward_gap = min(min_forward_gap, forward_gap)

            status = "CRUISING"
            if collisions:
                result["passed"] = (args.test == "collision" and
                                    any(c["actor_id"] == obstacle.id for c in collisions)
                                    and obstacle.id in seen and top_speed > 1.0)
                result["reason"] = "Collision sensor triggered"
                break
            if (args.test == "avoid" and shift_start is None and gap < 50
                    and obstacle.id in seen and progress < 15
                    and lane_clear(world, ego, origin, forward, right, progress, offset)):
                shift_start = progress + 2.0
                print(f"OBSTACLE DETECTED: requesting {direction} lane change")
            target_s = progress + max(4.0, speed * 0.8)
            target_y = lateral_target(target_s, shift_start, offset)
            target = carla.Location(x=origin.x + forward.x * target_s + right.x * target_y,
                                    y=origin.y + forward.y * target_s + right.y * target_y,
                                    z=transform.location.z)
            local = target - transform.location
            local_forward = dot_xy(local, transform.get_forward_vector())
            local_right = dot_xy(local, transform.get_right_vector())
            steer = steering_command(local_forward, local_right, wheelbase, steering_limit)
            desired_speed = 3.0 if args.test == "collision" else SPEED
            throttle, brake = speed_command(speed, desired_speed, gap, args.test != "collision")
            if shift_start is not None:
                status = "AVOIDING"
            if args.test != "collision" and gap < brake_distance(speed):
                throttle, brake, status = 0.0, 1.0, "BRAKING"
            ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
            detected = obstacle.id in seen
            stopped_for = stopped_for + STEP if (
                speed < 0.15 and top_speed > 1 and brake > 0.9 and detected) else 0.0
            if (args.test == "brake" and stopped_for >= 1
                    and 1 < forward_gap <= 15 and abs(lateral) < 0.75):
                result.update(passed=True, reason="Stopped before the obstacle without collision")
                break
            if (args.test == "avoid" and shift_start is not None and detected
                    and dot_xy(obstacle_delta, forward) < -8 and abs(lateral - offset) < 0.75):
                result.update(passed=True, reason="Passed the obstacle in the adjacent lane without collision")
                break
            spectator.set_transform(carla.Transform(
                carla.Location(x=transform.location.x - forward.x * 9,
                               y=transform.location.y - forward.y * 9,
                               z=transform.location.z + 6),
                carla.Rotation(pitch=-25, yaw=transform.rotation.yaw)))
            if index % 20 == 0:
                label = f"{gap:.1f} m" if math.isfinite(gap) else "clear"
                print(f"{status:10} speed={speed * 3.6:4.1f} km/h  sensor={label}")
            world.debug.draw_string(transform.location + carla.Location(z=2.5), status,
                                    color=carla.Color(255, 220, 0), life_time=STEP * 2)
            world.tick()
            time.sleep(max(0.0, STEP - (time.monotonic() - wall_start)))

        # Let any last sensor callbacks arrive before declaring success.
        ego.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(5):
            world.tick()
            time.sleep(STEP)
        _, seen, collisions = readings.snapshot(world.get_snapshot().timestamp.elapsed_seconds)
        if collisions and args.test != "collision":
            result.update(passed=False, reason="Collision recorded")
        result.update(obstacle_detected=obstacle.id in seen,
                      collision_events=collisions, peak_speed_kmh=round(top_speed * 3.6, 2),
                      lane_change_requested=shift_start is not None,
                      minimum_forward_gap_m=(round(min_forward_gap, 2)
                                             if math.isfinite(min_forward_gap) else None))
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
        for actor in reversed(actors):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        if changed_settings:
            world.apply_settings(settings)
        spectator.set_transform(saved_view)

    print("\n" + ("PASS" if result["passed"] else "FAIL") + ": " + result["reason"])
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("Result saved to", args.output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped; test actors and world settings have been cleaned up.")
    except RuntimeError as error:
        raise SystemExit(f"ERROR: {error}") from error
