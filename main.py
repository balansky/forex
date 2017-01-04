import argparse
import sys

def create_parser(prog):
    parser = argparse.ArgumentParser(prog=prog, allow_abbrev=False)
    parser.add_argument('command', type=str, help='monitor, skml, tsml')
    parser.add_argument('--target_pair','-p', type=str,default='EURUSD' ,help="Main Pair To Be Learned Or Predicted")
    parser.add_argument('--forward_days','-fw',type=int, default=5, help="Forward Days")
    parser.add_argument('--backward_days', '-bw', type=int, default=10, help="Backward Days")
    parser.add_argument('--feature_days','-f', type=int,default=60, help="Feature Days")
    parser.add_argument('--pairs','-ps', nargs='*', help="Forex Pairs Learned By Advanced Learner",
                        default=["EURUSD", "AUDUSD", "CHFJPY", "EURCHF", "EURGBP", "EURJPY","GBPCHF", "GBPJPY",
                                 "GBPUSD", "USDCAD","USDCHF","USDJPY"])
    return parser


def execute_from_command(argv):
    parser = create_parser(argv[0])
    args = parser.parse_args(argv[1:])
    command = args.command
    if command == "monitor":
        from core import monitor
        monitor.start(args)
    elif command == "skml":
        from core import skml
        skml.test(args)
    elif command == "tsml":
        from core import tsml
        tsml.test_tf()
    else:
        print("None")

if __name__=="__main__":
    execute_from_command(sys.argv)
