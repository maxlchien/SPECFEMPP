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

        # Create a reference node
        reference = nodes.reference("", "", internal=False, refuri=download_path)
        reference += nodes.Text(link_text)

        # Wrap in a paragraph
        paragraph = nodes.paragraph()
        paragraph += reference

        return [paragraph]


def setup(app):
    app.add_directive("download-folder", DownloadFolderDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
