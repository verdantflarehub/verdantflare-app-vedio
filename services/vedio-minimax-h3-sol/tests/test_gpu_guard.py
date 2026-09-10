import os
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_guard import verify_gpu_allocation


class GPUAllocationTest(TestCase):
    first = '00000000-0000-0000-0000-000000000001'
    second = '00000000-0000-0000-0000-000000000002'
    third = '00000000-0000-0000-0000-000000000003'

    def test_current_pair_accepts_uuid_prefix_and_order(self):
        with patch.dict(os.environ, SOL_EXPECTED_GPU_UUIDS=f'{self.first},GPU-{self.second}'):
            self.assertEqual(verify_gpu_allocation([self.second, 'GPU-'+self.first]), [self.second,self.first])

    def test_missing_allocation_fails_closed(self):
        with patch.dict(os.environ, SOL_EXPECTED_GPU_UUIDS=''):
            with self.assertRaises(ValueError):
                verify_gpu_allocation([self.first,self.second])

    def test_different_or_duplicate_visible_gpu_is_rejected(self):
        with patch.dict(os.environ, SOL_EXPECTED_GPU_UUIDS=f'{self.first},{self.second}'):
            for actual in ([self.first,self.third],[self.first,self.first],[self.first]):
                with self.assertRaises(ValueError):
                    verify_gpu_allocation(actual)

    def test_duplicate_or_malformed_approved_pair_is_rejected(self):
        for expected in (f'{self.first},{self.first}', 'invalid,invalid'):
            with patch.dict(os.environ, SOL_EXPECTED_GPU_UUIDS=expected):
                with self.assertRaises(ValueError):
                    verify_gpu_allocation([self.first,self.second])
