"""
dataset_names.py

This module defines lists of dataset names and functions to retrieve dataset lists based on command-line arguments.
"""

def get_database_list_from_arguments(sys_argv):
    """
    Retrieves a list of datasets based on command-line arguments.

    Parameters:
    - sys_argv (list): Command-line arguments.

    Returns:
    - list: A list of dataset names.
    """
    dataset_list = list1

    if len(sys_argv) == 2:
        arg = sys_argv[1]

        if ":" in arg:
            arg_arr = arg.split(":")
            if len(arg_arr == 3):
                increment = int(arg_arr[2])
            else:
                increment = 1
            start = int(arg_arr[0])
            end = int(arg_arr[1])
            dataset_list = dataset_list[start:end:increment]
        elif arg.isnumeric():
            index = int(arg)
            dataset_list = dataset_list[index:index + 1]
        else:
            # Dataset name is given
            index = dataset_list.index(arg)
            dataset_list = dataset_list[index:index + 1]

    if len(sys_argv) == 3:
        start = int(sys_argv[1])
        end = int(sys_argv[2])
        dataset_list = dataset_list[start:end]

    return dataset_list

# Lists of various dataset categories
list1 = ["Yoga"]
list2 = ["WordSynonyms", "Yoga"]

list_cr = ["CricketX", "CricketY", "CricketZ"]

list_pro = ["CricketX", "CricketY", "CricketZ",
            "UWaveGestureLibraryX", "UWaveGestureLibraryY",
            "UWaveGestureLibraryZ", "UWaveGestureLibraryAll"]

list5 = ["ChlorineConcentration", "Coffee", "ECG200", "GunPoint",
         "DistalPhalanxOutlineCorrect", "HandOutlines"]

list3 = ["Coffee", "ECG200", "GunPoint"]

list_2_class = ["BirdChicken", "Coffee", "DistalPhalanxOutlineCorrect", "Earthquakes",
                "ECG200", "ECGFiveDays", "GunPoint", "Ham", "HandOutlines", "Herring",
                "ItalyPowerDemand", "Lightning2", "MiddlePhalanxOutlineCorrect", "MoteStrain",
                "PhalangesOutlinesCorrect", "ProximalPhalanxOutlineCorrect", "ShapeletSim",
                "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "Strawberry", "ToeSegmentation1",
                "ToeSegmentation2", "TwoLeadECG", "Wafer", "Wine", "WormsTwoClass", "Yoga"]

list_image_datasets = ["DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
                       "DistalPhalanxTW", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish",
                       "HandOutlines", "Herring", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect",
                       "MiddlePhalanxTW", "OSULeaf", "PhalangesOutlinesCorrect", "ProximalPhalanxOutlineAgeGroup",
                       "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "ShapesAll", "SwedishLeaf", "Symbols",
                       "WordSynonyms", "Yoga"]

list_learning_shapelet_datasets = ["Adiac", "Beef", "BirdChicken", "ChlorineConcentration", "Coffee",
                                   "DiatomSizeReduction", "ECGFiveDays", "FaceFour", "ItalyPowerDemand",
                                   "Lightning7", "MedicalImages", "MoteStrain", "SonyAIBORobotSurface1",
                                   "SonyAIBORobotSurface2", "Symbols", "Trace", "TwoLeadECG", "CricketX",
                                   "CricketY", "CricketZ", "Lightning2", "Mallat", "Meat", "NonInvasiveFatalECGThorax1",
                                   "NonInvasiveFatalECGThorax2", "OliveOil", "ScreenType", "SmallKitchenAppliances",
                                   "StarlightCurves", "Worms", "WormsTwoClass", "Yoga"]

list_all_85 = ["Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF", "ChlorineConcentration",
               "CinCECGtorso", "Coffee", "Computers", "CricketX", "CricketY", "CricketZ", "DiatomSizeReduction",
               "DistalPhalanxOutlineCorrect", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "Earthquakes",
               "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords",
               "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines", "Haptics", "Herring", "InlineSkate",
               "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7",
               "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxOutlineAgeGroup",
               "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFatalECGThorax1", "NonInvasiveFatalECGThorax2", "OliveOil",
               "OSULeaf", "PhalangesOutlinesCorrect", "Plane", "ProximalPhalanxOutlineCorrect",
               "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType",
               "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2",
               "StarlightCurves", "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl", "ToeSegmentation1",
               "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UWaveGestureLibraryX", "UWaveGestureLibraryY",
               "UWaveGestureLibraryZ", "UWaveGestureLibraryAll", "Wafer", "Wine", "WordSynonyms", "Worms",
               "WormsTwoClass", "Yoga"]

def get_all_dataset_names():
    """
    Returns the list of all dataset names.

    Returns:
    - list: A list of all dataset names.
    """
    return list_all_85
