#!/usr/bin/env python

# This script is used in setup.cfg and circleci.

from subprocess import check_output


def _run(args):
    """Run command."""
    return check_output(args).decode("utf8").strip()


def get_version():
    """Define package version dynamically."""
    describe_cmd = ["git", "describe", "--tags", "--always"]
    last_tag = _run(describe_cmd + ["--abbrev=0"])
    # describe = '1.0.14' / '1.0.14-2-gfaa2442'  {tag}-{nb_commit_since_tag}-{hash}'
    describe = _run(describe_cmd)
    if describe == last_tag:
        return last_tag
    short_hash = describe[len(last_tag) + 1 :].split("-")[1]
    return "{}.dev.{}".format(last_tag, short_hash)


VERSION = get_version() or "local"

if __name__ == "__main__":
    print(VERSION)
