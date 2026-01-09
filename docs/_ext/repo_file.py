import subprocess
from docutils import nodes
from sphinx.util.docutils import SphinxRole


class RepoFileRole(SphinxRole):
    """Role for linking to files in the GitHub repository.

    Usage:
        :repo-file:`path/to/file.cpp`

    This will create a link to the file in the GitHub repository
    on the current git branch.
    """

    def run(self):
        # Access config values (github_url registered in conf.py)
        github_url = self.env.config.github_url
        git_branch = self.env.config.git_branch

        print(f"DEBUG: RepoFileRole using branch: {git_branch} for file: {self.text}")

        url = f"{github_url}/blob/{git_branch}/{self.text}"
        node = nodes.reference("", "", internal=False, refuri=url)
        node += nodes.literal(self.text, self.text)
        return [node], []


def get_git_branch(confdir):
    """Get the current git branch name."""
    try:
        git_branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=confdir,
                stderr=subprocess.STDOUT,
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to main if git command fails
        git_branch = "main"
    return git_branch


def setup(app):
    # Detect the current git branch
    git_branch = get_git_branch(app.confdir)

    print(f"DEBUG: Detected git branch: {git_branch}")

    # Register config value - can be overridden in conf.py
    app.add_config_value("git_branch", git_branch, "html")

    # Register the custom role
    app.add_role("repo-file", RepoFileRole())

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
