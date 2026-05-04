#!/usr/bin/env python3
"""Run ablation study: 5 agent variants compared by Elo."""
import sys
sys.path.insert(0, '/home/coder/nexus')

import argparse

ABLATION_CONFIGS = {
    'NEXUS-full':      {'restnet': True,  'gumbel': True,  'opp_gru': True,  'disco': True,  'multi_value': True},
    'NEXUS-no-trans':  {'restnet': False, 'gumbel': True,  'opp_gru': True,  'disco': True,  'multi_value': True},
    'NEXUS-no-gumbel': {'restnet': True,  'gumbel': False, 'opp_gru': True,  'disco': True,  'multi_value': True},
    'NEXUS-no-gru':    {'restnet': True,  'gumbel': True,  'opp_gru': False, 'disco': True,  'multi_value': True},
    'NEXUS-no-disco':  {'restnet': True,  'gumbel': True,  'opp_gru': True,  'disco': False, 'multi_value': True},
}


def main():
    parser = argparse.ArgumentParser(description='NEXUS v2 Ablation Study')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--eval-interval', type=int, default=50)
    args = parser.parse_args()

    print("Ablation study configurations:")
    for name, cfg in ABLATION_CONFIGS.items():
        disabled = [k for k, v in cfg.items() if not v]
        print(f"  {name}: disabled = {disabled or 'none'}")

    print(f"\nTo run: train each variant for {args.iterations} iterations,")
    print(f"evaluate every {args.eval_interval} iterations via round-robin tournament.")
    print("\nThis script is a template - each ablation variant requires modifying")
    print("the network/MCTS configuration before training. See the blueprint for details.")


if __name__ == '__main__':
    main()
