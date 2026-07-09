PRUNER_IMPLEMENTATIONS = {
    'mcprune': {
        'label': 'MCPrune',
        'status': 'proposed',
        'criterion': 'topology + grad_ema + selective_hvp',
    },
    'hap': {
        'label': 'Hessian-energy proxy',
        'status': 'proxy',
        'criterion': 'inverse layer HVP energy + channel HVP energy',
        'limitation': 'No neural implant stage from the HAP paper.',
    },
    'snows': {
        'label': 'HVP-energy proxy',
        'status': 'proxy',
        'criterion': 'channel HVP energy',
        'limitation': 'No SNOWS reconstruction objective or Newton weight update.',
    },
    'ato': {
        'label': 'L1-magnitude proxy',
        'status': 'proxy',
        'criterion': 'mean absolute channel weight',
        'limitation': 'No ATO controller network or pruning-from-scratch objective.',
    },
    'st': {
        'label': 'Weight-Grad-EMA proxy',
        'status': 'proxy',
        'criterion': 'mean absolute weight * sqrt(grad_ema)',
    },
    'dfpc': {
        'label': 'Filter-distance proxy',
        'status': 'proxy',
        'criterion': 'sum of pairwise filter distances',
        'limitation': 'No parameter compensation stage.',
    },
    'tpp': {
        'label': 'Weight-Grad-EMA proxy',
        'status': 'proxy',
        'criterion': 'mean absolute weight * sqrt(grad_ema)',
        'limitation': 'Currently identical to the ST proxy criterion.',
    },
}


def annotate_pruner_config(config):
    strategy = config.get('strategy', {})
    method = strategy.get('method', '').lower()
    if method != 'pdt':
        details = {
            'label': method.upper(),
            'status': 'configured',
            'criterion': strategy.get('type', method),
        }
        strategy['implementation'] = details
        return method, details
    pruner = strategy.get('pruner', 'mcprune').lower()
    if pruner not in PRUNER_IMPLEMENTATIONS:
        raise ValueError(f"Unknown pruning implementation: {pruner}")
    details = dict(PRUNER_IMPLEMENTATIONS[pruner])
    strategy['implementation'] = details
    return pruner, details
