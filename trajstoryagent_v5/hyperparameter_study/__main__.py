import sys
from ..ablation.experiment import main as _main
def main():
    return _main([*sys.argv[1:], "--variant", "hyperparameter"])

if __name__ == "__main__":
    main()
