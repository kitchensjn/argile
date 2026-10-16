import numpy as np
import pandas as pd


def _format_parameter_string(s):
    """Formats the parameter string to match expected time:value format

    Parameters
    ----------
    s : string
        Input parameter string

    Returns
    -------
    formatted : string
        Formatted parameter string
    """

    if ":" in s:
        split_string = s.replace(",", ":").split(":")
        epochs = np.array(split_string[::2]).astype(float)
        parameters = np.array(split_string[1::2])
        sorted_epochs = np.argsort(epochs)
        formatted = ""
        if epochs[sorted_epochs[0]] != 0:
            formatted += "0:0,"
        for e in sorted_epochs:
            formatted += f"{epochs[e]}:{parameters[e]},"
        return formatted[:-1]
    elif s == "nan":
        raise RuntimeError(f"String `{s}` cannot be formatted.")
    return f"0:{s}"


class WorldMap:
    """Stores a map of the demes

    Attributes
    ----------
    demes : pandas.DataFrame
        Information about demes, one row per deme. Mandatory columns are `id` and `suitability`, often includes `xcoord` and `ycoord` for plotting
    connections : pandas.DataFrame
        Information about connections between demes. Mandatory columns are `id`, `deme_0`, `deme_1`, and `migration_modifier`. ID in `deme_0` must be smaller than ID in `deme_1`
    samples : pandas.DataFrame
        Information about sample locations. Mandatory columns `id` and `deme`
    epochs : numpy.ndarray
        Increasing epoch timings (back in time) based on changes noted in the `demes.suitability` and `connections.migration_modifier` columns
    epoch_durations : numpy.ndarray
        Ordered epoch durations, built directly from `epochs`
    sample_locations : numpy.ndarray
        Probability of finding each sample in each deme (#samples x #demes)
    suitabilities : numpy.ndarray
        2D array which contains suitability information for all demes over time (#epochs x #demes). Order is the same as `epochs` and `demes.id`
    migration_modifiers : numpy.ndarray
        2D array which contains migration modifiers for all connections over time (#epochs x #connections). Order is the same as `epochs` and `connections.id`
    parameters : numpy.ndarray
        Available variables within the map, including the default migration rate ("coefficient") and any variables in the `migration.migration_modifier` column
    """

    def __init__(self, demes, connections, samples=None):
        """Initializes the WorldMap object

        Parameters
        ----------
        demes : pandas.DataFrame
            From demes.tsv file
        connections : pandas.DataFrame
            From connections.tsv file
        samples : pandas.DataFrame
            From samples.tsv file (optional)
        """

        self.parameters = ["coefficient"]

        self.demes = demes.copy()  
        self.demes["suitability"] = self.demes["suitability"].astype(str)
        formatted_suitability_strings = []
        for suitability in self.demes["suitability"]:
            formatted_suitability_strings.append(_format_parameter_string(suitability))
        self.demes["suitability"] = formatted_suitability_strings        

        self.connections = connections.copy()
        self.connections["migration_modifier"] = self.connections["migration_modifier"].astype(str)
        formatted_modifier_strings = []
        for modifier in self.connections["migration_modifier"]:
            formatted_modifier_strings.append(_format_parameter_string(modifier))
        self.connections["migration_modifier"] = formatted_modifier_strings
    
        epochs = [0]
        for suitability_string in self.demes["suitability"]:
            epoch_formatter = suitability_string.replace(",", ":").split(":")
            for epoch_assignment in epoch_formatter[::2]:
                epoch_assignment = int(float(epoch_assignment))
                if epoch_assignment not in epochs:
                    epochs.append(epoch_assignment)
        migration_modifier_variables = []
        for modifier_string in self.connections["migration_modifier"]:
            epoch_formatter = modifier_string.replace(",", ":").split(":")
            for t in range(0, len(epoch_formatter), 2):
                epoch_assignment = int(float(epoch_formatter[t]))
                if epoch_assignment not in epochs:
                    epochs.append(epoch_assignment)
        self.epochs = np.sort(epochs)
        self.epoch_durations = np.diff(self.epochs, append=(self.epochs[-1]-1))

        self.suitabilities = self._build_suitabilities_array()

        self.migration_modifiers = self._build_migration_modifiers_array()
        modifiers = np.unique(self.migration_modifiers)
        if len(modifiers) > 1:
            for mod in modifiers:
                if any(char.isalpha() for char in mod):
                    self.parameters.append(mod)
    
        self.samples = None
        self.sample_locations = None
        if samples is not None:
            self.samples = samples.copy()
            self.sample_locations = self._build_sample_locations_array()
    
    def get_deme_suitability_at_time(self, id, time):
        """Gets the type for specified deme

        Wrapper of pandas.DataFrame function. Assumes that deme IDs are unique.

        Parameters
        ----------
        id : int
            ID of deme
        time : int or float
            Specified time of map

        Returns
        -------
        deme_type : float
            Type of deme
        """

        deme_suitability = self.demes.loc[self.demes["id"]==id,"suitability"].iloc[0]
        epochs = deme_suitability.split(",")
        for t in range(len(epochs)-1):
            details = epochs[t].split(":")
            start = float(details[0])
            end = float(epochs[t+1].split(":")[0])
            if (time >= start) and (time < end):
                return float(details[1])
        return float(epochs[-1].split(":")[1])

    def get_all_deme_suitabilities_at_time(self, time):
        """

        Parameters
        ----------
        time : int or float
            Specified time of world map

        Returns
        -------
        suitabilities_at_time : pd.Series
            Deme types ordered by the demes dataframe
        """

        suitabilities_at_time = pd.Series([self.get_deme_suitability_at_time(id, time) for id in self.demes["id"]])
        return suitabilities_at_time

    def get_connection_migration_modifier_at_time(self, id, time):
        """Extracts the migration modifier for a connection at time

        Parameters
        ----------
        id : int
        time : int or float

        Returns
        -------
        migration_modifier : str
            Modifier of connection at specified time
        """
        connection_parameter = self.connections.loc[self.connections["id"]==id,"migration_modifier"].iloc[0]
        epochs = connection_parameter.split(",")
        for t in range(len(epochs)-1):
            details = epochs[t].split(":")
            start = float(details[0])
            end = float(epochs[t+1].split(":")[0])
            if (time >= start) and (time < end):
                return details[1]
        migration_modifier = epochs[-1].split(":")[1]
        return migration_modifier

    def get_all_connection_migration_modifiers_at_time(self, time):
        return pd.Series([self.get_connection_migration_modifier_at_time(id, time) for id in self.connections["id"]])

    def _build_sample_locations_array(self):
        """Formats the sample locations into probability distribution vectors

        Returns
        -------
        sample_locations_array : numpy.ndarray
            Probability distribution vector for each sample location (generally 0 in all demes except one)
        """

        if (self.samples is None):
            raise RuntimeError("No samples provided. Check that you've added samples to your WorldMap.")
        sample_locations_array = np.zeros((len(self.samples), len(self.demes)), dtype="float64")
        for i, sample in self.samples.iterrows():
            sample_locations_array[i,self.demes.loc[self.demes["id"]==sample["deme"]].index[0]] = 1
        return sample_locations_array

    def _build_suitabilities_array(self):
        """Builds the suitability array described in class attributes

        Returns
        -------
        suitabilities
        """

        suitabilities = np.zeros((len(self.epochs), len(self.demes)), dtype="float")
        for i,epoch in enumerate(self.epochs):
            suitabilities[i] = self.get_all_deme_suitabilities_at_time(epoch)
        return suitabilities
    
    def _build_migration_modifiers_array(self):
        """Builds the migration modifier array described in class attributes

        Returns
        -------
        modifiers
        """

        modifiers = np.zeros((len(self.epochs), len(self.connections)), dtype="object")
        for i,epoch in enumerate(self.epochs):
            modifiers[i] = self.get_all_connection_migration_modifiers_at_time(epoch)
        return modifiers