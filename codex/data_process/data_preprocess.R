require(magrittr)

##################################################
# DAPI Normalization #####
##################################################

#' Marker Normalization by Core
#'
#' This function normalizes the specified markers (`marker_to_norm`) by dividing them by the stable marker (`marker_stable`)
#' within each group defined by `core`. The normalization can be done using either the median or mean of the stable marker.
#'
#' @param df (data.frame) A data frame that contains the core identifiers and the markers to normalize.
#' @param core (string) The column name representing the core identifier by which the data is grouped.
#' @param marker_to_norm (string vector) A vector of column names representing the markers to normalize.
#' @param marker_stable (string) The column name of the stable marker used for normalization.
#' @param norm_method (string) The normalization method, either "median" or "mean". Default is "median".
#'
#' @return A data frame with the normalized marker values for each group defined by `core`. The specified markers will be
#'         divided by the median or mean of the stable marker within each group.
#'
#' @export
#'
#' @examples
#' # Example usage:
#' df <- data.frame(
#'     core = c("A", "A", "B", "B"),
#'     marker1 = c(10, 20, 30, 40),
#'     marker2 = c(15, 25, 35, 45),
#'     marker3 = c(5, 5, 10, 10)
#' )
#'
#' # Normalize marker1 and marker2 by stable_marker using the median
#' df_normalized <- df %>%
#'     marker_normalization_by_core("core", c("marker1", "marker2"), "marker3", "median")
marker_normalization_by_core <- function(df,
                                         core,
                                         marker_to_norm,
                                         marker_stable = "DAPI",
                                         norm_method = "median") {
    norm_fun <- list("median" = median, "mean" = mean)
    norm_fun <- norm_fun[[norm_method]]
    df_norm <- df %>%
        dplyr::group_by(dplyr::across(dplyr::all_of(core))) %>%
        dplyr::mutate(dplyr::across(dplyr::all_of(marker_to_norm), ~ .x / norm_fun(.data[[marker_stable]]))) %>%
        dplyr::ungroup()
    return(df_norm)
}


##################################################
# Marker Scaling #####
##################################################

#' Marker Scaling by Core
#'
#' This function scales the specified markers (`marker_to_scale`) within each group defined by `core`. 
#' Scaling is done by applying min-max normalization, which transforms the values in each marker to 
#' the range [0, 1] within each group of the core.
#'
#' @param df (data.frame) A data frame that contains the core identifiers and the markers to scale.
#' @param core (string) The column name representing the core identifier by which the data is grouped.
#' @param marker_to_scale (string vector) A vector of column names representing the markers to scale.
#'
#' @return A data frame with the scaled marker values for each group defined by `core`. 
#'         The specified markers will be scaled using min-max normalization within each group.
#'
#' @export
#'
#' @examples
#' # Example usage:
#' df <- data.frame(
#'     core = c("A", "A", "B", "B"),
#'     marker1 = c(10, 20, 30, 40),
#'     marker2 = c(15, 25, 35, 45)
#' )
#'
#' # Scale marker1 and marker2 using min-max scaling within each core group
#' df_scaled <- df %>%
#'     marker_scale_by_core("core", c("marker1", "marker2"))
marker_scale_by_core <- function(df, core, marker_to_scale) {
    df_scale <- df %>%
        dplyr::group_by(dplyr::across(dplyr::all_of(core))) %>%
        dplyr::mutate(dplyr::across(dplyr::all_of(marker_to_scale), ~ (.x - min(.x)) / (max(.x) - min(.x)))) %>%
        dplyr::ungroup()
    return(df_scale)
}


#' Marker Scaling Globally
#'
#' This function scales the specified markers (`marker_to_scale`) across the entire dataset.
#' Scaling is done by applying min-max normalization, which transforms the values 
#' of each marker to the range [0, 1] globally across the entire dataset.
#'
#' @param df (data.frame) A data frame that contains the markers to scale.
#' @param marker_to_scale (string vector) A vector of column names representing the markers to scale.
#'
#' @return A data frame with the globally scaled marker values. 
#'         The specified markers will be scaled using min-max normalization globally across the entire dataset.
#'
#' @export
#'
#' @examples
#' # Example usage:
#' df <- data.frame(
#'     marker1 = c(10, 20, 30, 40),
#'     marker2 = c(15, 25, 35, 45)
#' )
#'
#' # Scale marker1 and marker2 using global min-max scaling
#' df_scaled <- df %>%
#'     marker_scale_globally(c("marker1", "marker2"))
marker_scale_globally <- function(df, marker_to_scale) {
    df_scale <- df %>%
        dplyr::mutate(dplyr::across(dplyr::all_of(marker_to_scale), ~ (.x - min(.x)) / (max(.x) - min(.x)))) 
    return(df_scale)
}





