from pycodex.io import (
    summary_dir,
    organize_metadata_fusion,
    organize_metadata_keyence,
    summary_metadata,
)
import pandas as pd
import tifffile


class Marker:
    def __init__(self, marker_dir):
        self.marker_dir = marker_dir

    def summary_dir(self):
        """
        Summarize the contents of the marker directory.
        """
        summary_dir(self.marker_dir, indent="    ")

    def organize_metadata(
        self,
        platform: str,  # "keyence", "fusion"
        subfolders: bool = True,
        extensions: list[str] = [".tiff", ".tif"],
    ):
        """
        Organize metadata from marker files.
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
            raise ValueError(
                "Invalid platform. Options are 'keyence' or 'fusion'."
            )

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
        """
        df_metadata = self.metadata[region]
        df_metadata = df_metadata[
            df_metadata["marker"].isin(marker_list)
        ].reset_index(drop=True)

        marker_dict = {
            row["marker"]: tifffile.imread(row["path"])
            for _, row in df_metadata.iterrows()
        }
        return marker_dict
