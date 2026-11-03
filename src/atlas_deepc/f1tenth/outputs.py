"""Shared vehicle angle convention."""
import numpy as np

def wrap_angle(angle):
    return (np.asarray(angle, dtype=float) + np.pi) % (2*np.pi) - np.pi

