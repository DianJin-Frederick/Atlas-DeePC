"""Regression checks for commanded-versus-applied input bookkeeping."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from atlas_deepc.f1tenth import task


class AppliedInputsTest(unittest.TestCase):
    def transition(self, speed, acceleration):
        state=np.array([0.,0.,0.,speed,.2,0.,0.])
        command=np.array([acceleration,.04])
        ordinary=task.step(state,command)
        nxt,applied,internal=task.step(state,command,return_applied=True)
        np.testing.assert_array_equal(nxt,ordinary)
        np.testing.assert_allclose(applied,internal.mean(axis=0),rtol=0,atol=0)
        np.testing.assert_allclose(nxt[3]-state[3],.1*applied[0],rtol=0,atol=1e-12)
        np.testing.assert_allclose(internal[:,1],command[1],rtol=0,atol=0)
        np.testing.assert_array_equal(command,[acceleration,.04])
        return applied,internal

    def test_unsaturated_input_is_preserved(self):
        u,_=self.transition(6.,.8)
        np.testing.assert_allclose(u,[.8,.04],rtol=0,atol=1e-14)

    def test_positive_acceleration_suppressed_at_speed_limit(self):
        u,internal=self.transition(8.01,.8)
        self.assertEqual(u[0],0.)
        np.testing.assert_array_equal(internal[:,0],0.)

    def test_braking_at_speed_limit_is_preserved(self):
        u,_=self.transition(8.01,-1.)
        self.assertAlmostEqual(u[0],-1.)

    def test_limiting_within_interval_requires_average(self):
        u,internal=self.transition(7.99,1.)
        self.assertGreater(u[0],0.)
        self.assertLess(u[0],1.)
        self.assertGreater(np.ptp(internal[:,0]),0.)

    def test_speed_dependent_acceleration_bound(self):
        u,_=self.transition(7.5,3.)
        self.assertGreater(u[0],0.)
        self.assertLess(u[0],3.)

    def test_offline_outputs_unchanged_and_applied_inputs_consistent(self):
        source=task.collect_commanded_clean();applied=task.collect_clean()
        for key in ('X','Y','dt'):np.testing.assert_array_equal(source[key],applied[key])
        np.testing.assert_array_equal(source['U'],applied['U_commanded'])
        np.testing.assert_array_equal(applied['U'],applied['U_applied'])
        np.testing.assert_allclose(applied['X_next'][...,3]-applied['X'][...,3],.1*applied['U'][...,0],rtol=0,atol=1e-12)
        np.testing.assert_allclose(applied['U'],applied['applied_substeps'].mean(axis=2),rtol=0,atol=0)
        self.assertGreater(np.count_nonzero(abs(applied['U'][...,0]-source['U'][...,0])>1e-10),0)

    def test_offline_measurement_noise_preserves_applied_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            noisy=task.collect(Path(tmp)/'noisy.npz',sigma=.001)
        clean=task.collect_clean()
        np.testing.assert_array_equal(noisy['U'],clean['U'])
        np.testing.assert_array_equal(noisy['U_commanded'],clean['U_commanded'])
        self.assertGreater(np.max(abs(noisy['Y']-clean['Y'])),0.)
        self.assertEqual(str(noisy['input_history_convention']),task.INPUT_HISTORY_CONVENTION)

if __name__=='__main__':unittest.main()
