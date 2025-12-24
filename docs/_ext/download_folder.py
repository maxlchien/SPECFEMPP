import os
import shutil
from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
from sphinx.util import logging

logger = logging.getLogger(__name__)


class DownloadFolderDirective(SphinxDirective):
    """
    Directive to download an entire folder as a zip file.

    Usage:
        .. download-folder:: path/to/folder
           :filename: custom_name.zip
           :text: Download my folder
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        "filename": directives.unchanged,
        "text": directives.unchanged,
    }

    def run(self):
        env = self.env
        folder_path = self.arguments[0]

        # Resolve the folder path relative to the source directory
        source_dir = os.path.dirname(env.doc2path(env.docname))
        abs_folder_path = os.path.normpath(os.path.join(source_dir, folder_path))

        if not os.path.isdir(abs_folder_path):
            logger.warning(
                f"Folder not found: {abs_folder_path}", location=self.get_location()
            )
            return []

        # Determine the zip filename
        if "filename" in self.options:
            zip_filename = self.options["filename"]
            if not zip_filename.endswith(".zip"):
                zip_filename += ".zip"
        else:
            folder_name = os.path.basename(abs_folder_path.rstrip(os.sep))
            zip_filename = f"{folder_name}.zip"

        # Create zip in the same directory as the output HTML file
        # Get the output directory for the current document
        docname_path = env.docname.replace(os.sep, "/")
        output_doc_dir = os.path.join(env.app.outdir, os.path.dirname(docname_path))
        os.makedirs(output_doc_dir, exist_ok=True)

        zip_path = os.path.join(output_doc_dir, zip_filename)
        zip_path_without_ext = zip_path.replace(".zip", "")

        # Create the zip file
        try:
            shutil.make_archive(zip_path_without_ext, "zip", abs_folder_path)
            logger.info(f"Created zip: {zip_path}")
        except Exception as e:
            logger.warning(
                f"Failed to create zip for {folder_path}: {e}",
                location=self.get_location(),
            )
            return []

        # Determine link text
        link_text = self.options.get(
            "text", f"Download {os.path.basename(folder_path)}"
        )

        # Simple relative path - just the filename since it's in the same directory
        download_path = zip_filename

        # Create a reference node for the download link (keep simple relative path)
        reference = nodes.reference("", "", internal=False, refuri=download_path)
        reference += nodes.Text(f"{link_text} here")

        # Wrap link in a paragraph with "Download " prefix
        link_paragraph = nodes.paragraph()
        link_paragraph += nodes.Text("Download ")
        link_paragraph += reference

        # Build full URL for curl/wget commands
        # Get ReadTheDocs environment variables
        version = os.environ.get("READTHEDOCS_VERSION")
        version_type = os.environ.get("READTHEDOCS_VERSION_TYPE")
        project_slug = os.environ.get("READTHEDOCS_PROJECT", "specfem2d-kokkos")

        # Determine base URL based on build type
        if version_type == "external":
            # PR build - use the special PR build URL format
            base_url = f"https://{project_slug}--{version}.org.readthedocs.build"
        else:
            # Regular build (branch, tag, or latest)
            base_url = getattr(env.config, "html_baseurl", "").rstrip("/")
            if not base_url:
                base_url = f"https://{project_slug}.readthedocs.io"

        # Fallback version if not on ReadTheDocs
        if not version:
            version = getattr(env.config, "version", "latest")

        # Construct full URL for curl/wget instructions
        doc_dir = os.path.dirname(docname_path)
        if doc_dir:
            full_url = f"{base_url}/en/{version}/{doc_dir}/{zip_filename}"
        else:
            full_url = f"{base_url}/en/{version}/{zip_filename}"

        # Add instructions paragraph
        instructions = nodes.paragraph()
        instructions += nodes.Text("Or download using command line:")

        # Create code block with curl and wget commands using full URL
        curl_command = f"curl -O {full_url}"
        wget_command = f"wget {full_url}"
        code_text = f"# Using curl:\n{curl_command}\n\n# Using wget:\n{wget_command}"

        code_block = nodes.literal_block(code_text, code_text)
        code_block["language"] = "bash"

        return [link_paragraph, instructions, code_block]


def setup(app):
    app.add_directive("download-folder", DownloadFolderDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
