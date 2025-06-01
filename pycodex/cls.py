import pandas as pd
import tifffile
from IPython.display import display

from pycodex.io import (
    organize_metadata_fusion,
    organize_metadata_keyence,
    summary_dir,
    summary_metadata,
)


class MarkerMetadata:
    def __init__(self, marker_dir):
        self.marker_dir = marker_dir

    @staticmethod
    def display_items(items: list[str], ncol: int = 10):
        """
        Display a list in tabular format.

        Args:
            marker_list (dict): Dictionary or list of markers to display in tabular form.
            ncol (int): Number of columns to display in the output table.

        Returns:
            None: This function displays the DataFrame of markers.
        """
        ncol = min(ncol, len(items))
        markers_df = pd.DataFrame(
            [items[i : i + ncol] for i in range(0, len(items), ncol)],
            columns=[i + 1 for i in range(ncol)],
        ).fillna("")

        display(markers_df)

    def summary_dir(self):
        """
        Summarize the contents of the marker directory.
        """
        summary_dir(self.marker_dir, indent="    ")

    def organize_metadata(
        self,
        platform: str,  # "keyence", "fusion"
        subfolders: bool = True,
        extensions: list[str] = [".tiff", ".tif", ".ome.tiff"],
    ):
        """
        Organize metadata from marker files.

        Parameters
        ----------
        platform : str
            Platform used to acquire marker images.
            Options are "keyence" or "fusion".
        subfolders : bool, optional
            Search for markers in subfolders, by default True.
        extensions : list[str], optional
            File extensions to search for. Default is [".tiff", ".tif"].
        """
        # Define function to parse marker information
        if platform == "fusion":
            self.metadata = organize_metadata_fusion(
                self.marker_dir, subfolders, extensions
            )
        elif platform == "keyence":
            self.metadata = organize_metadata_keyence(
                self.marker_dir, subfolders, extensions
            )
        else:
            raise ValueError("Invalid platform. Options are 'keyence' or 'fusion'.")

    def summary_metadata(self):
        """
        Summarize marker information across regions.
        """
        (
            self.regions,
            self.unique_markers,
            self.blank_markers,
            self.missing_markers,
        ) = summary_metadata(self.metadata, indent="    ")

    def organize_marker_dict(
        self, region: str, marker_list: list[str]
    ) -> dict[str, pd.DataFrame]:
        """
        Organize marker dictionary for a specific region.

        Parameters
        ----------
        region : str
            Region to organize.
        marker_list : list[str]
            List of markers to organize.
        """
        df_metadata = self.metadata[region]
        df_metadata = df_metadata[df_metadata["marker"].isin(marker_list)].reset_index(
            drop=True
        )

        marker_dict = {
            row["marker"]: tifffile.imread(row["path"])
            for _, row in df_metadata.iterrows()
        }
        return marker_dict
