"""Production entrypoint: select the D-root log directory before importing text/audio modules."""
from __future__ import annotations
import argparse,os
from pathlib import Path
from scripts.production.state.ledger import ROOT_DEFAULT

def configure_logs(argv=None):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument("--root",default=str(ROOT_DEFAULT))
    args,_=parser.parse_known_args(argv)
    root=Path(args.root).resolve()
    os.environ.setdefault("DUPLEX_LOGS_DIR",str(root/"logs"))
    return Path(os.environ["DUPLEX_LOGS_DIR"])

def main():
    configure_logs()
    from scripts.production.orchestration.run import main as run_main
    return run_main()

if __name__=="__main__":main()
