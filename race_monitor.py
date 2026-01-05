import time
from collections import defaultdict

class RaceMonitor:
    def __init__(self):
        self.car_stats = {}
        self.lap_times = defaultdict(list)
        self.wall_hits = defaultdict(int)
        self.checkpoint_times = defaultdict(list)
        self.start_times = {}
        
    def track_car(self, car):
        if car not in self.car_stats:
            self.car_stats[car] = {
                'lap_count': 0,
                'last_checkpoint': -1,
                'wall_hits': 0,
                'total_distance': 0,
                'start_time': time.time()
            }
            self.start_times[car] = time.time()
    
    def update(self, car):
        if car not in self.car_stats:
            self.track_car(car)
            
        stats = self.car_stats[car]
        
        # Track wall hits
        if car.collisions > stats['wall_hits']:
            self.wall_hits[car] += 1
            stats['wall_hits'] = car.collisions
        
        # Track checkpoints
        if car.last_checkpoint != stats['last_checkpoint']:
            checkpoint_time = time.time() - stats['start_time']
            self.checkpoint_times[car].append(checkpoint_time)
            stats['last_checkpoint'] = car.last_checkpoint
        
        # Track laps
        if car.laps_completed > stats['lap_count']:
            lap_time = time.time() - stats['start_time']
            self.lap_times[car].append(lap_time)
            stats['lap_count'] = car.laps_completed
            stats['start_time'] = time.time()
        
        # Track distance
        stats['total_distance'] = car.distance_traveled
    
    def get_stats(self, car):
        if car not in self.car_stats:
            return "Car not tracked"
            
        stats = self.car_stats[car]
        output = [
            f"Car Statistics:",
            f"- Laps Completed: {stats['lap_count']}",
            f"- Wall Hits: {stats['wall_hits']}",
            f"- Total Distance: {stats['total_distance']:.2f} units"
        ]
        
        if stats['lap_count'] > 0:
            output.append("- Lap Times:")
            for i, lap in enumerate(self.lap_times[car], 1):
                output.append(f"  Lap {i}: {lap:.2f}s")
        
        if self.checkpoint_times[car]:
            output.append("- Checkpoint Times:")
            for i, checkpoint in enumerate(self.checkpoint_times[car], 1):
                output.append(f"  Checkpoint {i}: {checkpoint:.2f}s")
        
        return "\n".join(output)