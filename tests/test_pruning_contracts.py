import unittest
from collections import OrderedDict
from types import SimpleNamespace

import torch

from src.pruning.pdt_strategies import PDTPruner
from src.pruning.registry import annotate_pruner_config


class PruningContractTest(unittest.TestCase):
    def test_proxy_status_is_explicit(self):
        for name in ('hap', 'snows', 'ato', 'st', 'dfpc', 'tpp'):
            config = {'strategy': {'method': 'pdt', 'pruner': name}}
            selected, details = annotate_pruner_config(config)
            self.assertEqual(selected, name)
            self.assertEqual(details['status'], 'proxy')

    def test_global_ranking_handles_underscore_names_and_survival_floor(self):
        model = torch.nn.Sequential(OrderedDict([
            ('conv_stem', torch.nn.Conv2d(3, 4, kernel_size=1)),
        ]))
        config = {'strategy': {
            'pruning_ratio': 0.9,
            'min_survival_ratio': 0.5,
            'group_selection_ratio': 1.0,
        }}
        pruner = PDTPruner(
            model, config, SimpleNamespace(), topology_groups=[['conv_stem']]
        )
        pruner.scheduled_pruning_progress = 1.0
        metadata = [({'names': ['conv_stem']}, index) for index in range(4)]

        pruner._global_rank_prune(
            scores=[1.0, 2.0, 3.0, 4.0],
            metadata=metadata,
            total_epochs=1,
            epoch=1,
            method_name='TEST',
        )

        self.assertEqual(int(model.conv_stem.mask.sum().item()), 2)


if __name__ == '__main__':
    unittest.main()
