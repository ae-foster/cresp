import yaml
import subprocess
import os
import errno
import sys
import logging
import inspect
from pathlib import Path

log = logging.getLogger(__name__)


def save_reproduce(argv, seed, run_path, git_hash, git_diff):
    """ create a file that can be used to reproduce the job """
    git_command = "git checkout " + git_hash + "&& "
    file_name = inspect.stack()[1][1]
    with open(".hydra/overrides.yaml", "r") as stream:
        try:
            overrides = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            log.info(exc)
    full_command = "python {} -m {}".format(file_name, " ".join(overrides))
    with open("{}/reproduce.sh".format(run_path), "w") as fp:
        fp.write(git_command + full_command)
    with open("{}/git_hash.txt".format(run_path), "w") as fp:
        fp.write(git_hash)
    with open("{}/git_diff.txt".format(run_path), "w") as fp:
        fp.write(git_diff)


def manage_exisiting_config(run_path, git_hash, resume=True, ckpt_name="last"):
    """ check whether the same config has already been ran """
    # if same config has been successful then skip
    success_path = os.path.join(run_path, "success.txt")
    has_success = os.path.isfile(success_path)
    ckpt_path = os.path.join(run_path, "ckpt/{}.ckpt".format(ckpt_name))
    has_ckpt = os.path.isfile(ckpt_path)

    if has_success:
        with open("{}/git_hash.txt".format(run_path), "r") as f:
            saved_git_hash = f.readline()
            same_hash = git_hash == saved_git_hash
        if same_hash and resume != "force":
            log.info("Skip: config already exists for this commit...")
            sys.exit()

    if not resume:  # if we do not want to resume return None for ckpt_path
        if has_ckpt:
            log.info("Not loading existing model because resume if False...")
        return success_path, None

    # if same config exists but has not been successful then restart from checkpoint
    if has_ckpt:
        with open("{}/git_hash.txt".format(run_path), "r") as f:
            saved_git_hash = f.readline()
            same_hash = git_hash == saved_git_hash
        if same_hash or resume == "force":
            log.info("Loading model from checkpoint...")
        else:
            return success_path, None
    else:
        ckpt_path = None
    return success_path, ckpt_path
