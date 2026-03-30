import time
import pytest
from weathergen.utils.profiler import Profiler

def test_profiler_basic():
    p = Profiler(enabled=True)
    with p.time_block("test"):
        time.sleep(0.1)
    
    stats = p.get_stats()
    assert "test" in stats
    assert stats["test"]["count"] == 1
    assert stats["test"]["total_time"] >= 0.1
    assert stats["test"]["avg_time"] >= 0.1

def test_profiler_disabled():
    p = Profiler(enabled=False)
    with p.time_block("test"):
        time.sleep(0.1)
    
    stats = p.get_stats()
    assert stats == {}

def test_profiler_accumulation():
    p = Profiler(enabled=True)
    for _ in range(3):
        with p.time_block("loop"):
            time.sleep(0.05)
    
    stats = p.get_stats()
    assert stats["loop"]["count"] == 3
    assert stats["loop"]["total_time"] >= 0.15

def test_profiler_reset():
    p = Profiler(enabled=True)
    with p.time_block("test"):
        time.sleep(0.1)
    p.reset()
    stats = p.get_stats()
    assert stats == {}
