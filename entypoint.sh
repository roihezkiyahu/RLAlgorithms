#!/bin/sh
cd /RLAlgorithms
git pull
cd ..
exec python -m RLAlgorithms.trainer.task "$@"