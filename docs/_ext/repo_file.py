import os
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
        git_ref = self.env.config.git_ref
        use_tree = self.env.config.use_tree

        print(
            f"DEBUG: RepoFileRole using ref: {git_ref} (use_tree: {use_tree}) for file: {self.text}"
        )

        # For PR builds and tag builds, use /tree/ with commit hash or tag name
        # For branch builds, use /blob/ with branch name
        path_type = "tree" if use_tree else "blob"
        url = f"{github_url}/{path_type}/{git_ref}/{self.text}"
        node = nodes.reference("", "", internal=False, refuri=url)
        node += nodes.literal(self.text, self.text)
        return [node], []


def get_git_ref_info(confdir):
    """Get git reference information for linking to GitHub.

    Handles both local development and ReadTheDocs builds.

    Returns:
        tuple: (git_ref, use_tree) where:
            - git_ref: branch name, tag name, or commit hash
            - use_tree: True if should use /tree/ in URL, False for /blob/

    On ReadTheDocs:
        - For PR builds (READTHEDOCS_VERSION_TYPE="external"):
          Uses READTHEDOCS_GIT_COMMIT_HASH for the commit SHA, use_tree=True
        - For tag builds (READTHEDOCS_VERSION_TYPE="tag"):
          Uses READTHEDOCS_GIT_IDENTIFIER for tag name, use_tree=True
        - For branch builds (READTHEDOCS_VERSION_TYPE="branch"):
          Uses READTHEDOCS_GIT_IDENTIFIER for branch name, use_tree=False

    Locally: Uses git command to get current branch, use_tree=False
    """
    # Check if we're on ReadTheDocs
    rtd_version_type = os.environ.get("READTHEDOCS_VERSION_TYPE")

    if rtd_version_type:
        # On ReadTheDocs
        is_pr_build = rtd_version_type == "external"
        is_tag_build = rtd_version_type == "tag"

        if is_pr_build:
            # For PR builds, use the commit hash with /tree/
            git_ref = os.environ.get("READTHEDOCS_GIT_COMMIT_HASH", "main")
            use_tree = True
        elif is_tag_build:
            # For tag builds, use the tag name with /tree/
            git_ref = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")
            use_tree = True
        else:
            # For branch builds, use the branch name with /blob/
            git_ref = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")
            use_tree = False

        return git_ref, use_tree

    # Local development: try to get branch from git
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
        # If in detached HEAD state, fall back to main
        if git_branch == "HEAD":
            git_branch = "main"
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to main if git command fails
        git_branch = "main"

    return git_branch, False


def setup(app):
    # Detect the git reference and build type
    git_ref, use_tree = get_git_ref_info(app.confdir)

    print(f"DEBUG: Detected git ref: {git_ref} (use_tree: {use_tree})")

    # Register config values - can be overridden in conf.py
    app.add_config_value("git_ref", git_ref, "html")
    app.add_config_value("use_tree", use_tree, "html")

    # Register the custom role
    app.add_role("repo-file", RepoFileRole())

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
