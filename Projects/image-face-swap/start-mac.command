#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "请先执行：python3 -m venv .venv，然后 .venv/bin/python -m pip install -r requirements.txt"
  read -r "reply?按回车退出"
  exit 1
fi
exec .venv/bin/python run.py --open-browser "$@"
