import threading
import time
import random

class RateLimiter:
    def __init__(self, max_calls, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
        
    def __call__(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                print(f"Sleeping for {sleep_time} seconds")
                if sleep_time > 0:
                    jitter = random.uniform(0, 0.1 * sleep_time)
                    print(f"Jittering for {jitter} seconds")
                    time.sleep(sleep_time + jitter)
                    now = time.time()
                self.calls = self.calls[1:]
                
            self.calls.append(now)

def is_rate_limit_error(error_msg):
    """Check if an error message indicates a rate limit issue"""
    rate_limit_indicators = [
        "rate limit", "ratelimit", "too many requests", 
        "429", "throttl", "quota exceeded", "limit exceeded", "overloaded", "quota", "limit",
        "current limit", "current quota", "current usage", "unavailable", "later", "try again"
    ]
    error_lower = str(error_msg).lower()
    return any(indicator in error_lower for indicator in rate_limit_indicators)