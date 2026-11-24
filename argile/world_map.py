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

    def __init__(self, demes, connections, samples=None, interpolate_suitability=False):
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
    
        self.suitability_epochs = [0]
        for suitability_string in self.demes["suitability"]:
            epoch_formatter = suitability_string.replace(",", ":").split(":")
            for epoch_assignment in epoch_formatter[::2]:
                epoch_assignment = int(float(epoch_assignment))
                if epoch_assignment not in self.suitability_epochs:
                    self.suitability_epochs.append(epoch_assignment)
        self.suitability_epochs = np.sort(self.suitability_epochs)

        self.suitabilities = self._build_suitabilities_array()

        self.migration_modifier_epochs = [0]
        migration_modifier_variables = []
        for modifier_string in self.connections["migration_modifier"]:
            epoch_formatter = modifier_string.replace(",", ":").split(":")
            for t in range(0, len(epoch_formatter), 2):
                epoch_assignment = int(float(epoch_formatter[t]))
                if epoch_assignment not in self.migration_modifier_epochs:
                    self.migration_modifier_epochs.append(epoch_assignment)
        self.migration_modifier_epochs = np.sort(self.migration_modifier_epochs)

        self.migration_modifiers = self._build_migration_modifiers_array()

        self.epochs = np.concatenate((self.suitability_epochs, self.migration_modifier_epochs))
        self.epochs = np.unique(self.epochs)
        self.epoch_durations = np.diff(self.epochs, append=(self.epochs[-1]-1))

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
    
    def _build_suitabilities_array(self):
        """Builds the suitability array described in class attributes

        Returns
        -------
        suitabilities : np.array
            Suitabilities of each deme in each suitability epoch
        """

        suitabilities = np.zeros((len(self.suitability_epochs), len(self.demes)), dtype="float")
        for j, suitability_string in enumerate(self.demes["suitability"]):
            suitability_string = suitability_string.split(",")
            for e in range(len(suitability_string)):
                time, suit = suitability_string[e].split(":")
                time = float(time)
                suit = float(suit)
                if (len(suitability_string) > (e+1)):
                    next_time = float(suitability_string[e+1].split(":")[0])
                    relevant_epochs = np.where((self.suitability_epochs>=time)&(self.suitability_epochs<next_time))[0]
                else:
                    relevant_epochs = np.where(self.suitability_epochs>=time)[0]
                suitabilities[relevant_epochs, j] = suit
        return suitabilities

    def get_deme_suitability_at_time(self, id, time):
        """Gets the suitability of a deme at a specified time

        Parameters
        ----------
        id : int
            ID of deme
        time : int or float
            Specified time of map

        Returns
        -------
        suitability : float
            Suitability of deme at time
        """

        deme_index = np.where(self.demes["id"]==id)[0][0]
        epoch_index = np.where(self.suitability_epochs<=time)[0][-1]
        suitability = self.suitabilities[epoch_index][deme_index]
        return suitability
    
    def get_all_deme_suitabilities_at_time(self, time):
        """Gets the suitabilities of all demes at a specified time

        Parameters
        ----------
        time : int or float
            Specified time of world map

        Returns
        -------
        suitabilities_at_time : pd.Series
            Deme types ordered by the demes dataframe
        """

        epoch_index = np.where(self.suitability_epochs<=time)[0][-1]
        suitabilities = self.suitabilities[epoch_index]
        return suitabilities

    def get_connection_migration_modifier_at_time(self, id, time):
        connection_index = np.where(self.connections["id"]==id)[0][0]
        epoch_index = np.where(self.migration_modifier_epochs<=time)[0][-1]
        migration_modifier = self.migration_modifier[epoch_index][connection_index]
        return migration_modifier

    def get_all_connection_migration_modifiers_at_time(self, time):
        epoch_index = np.where(self.migration_modifier_epochs<=time)[0][-1]
        migration_modifiers = self.migration_modifier[epoch_index]
        return migration_modifiers

    def _build_migration_modifiers_array(self):
        """Builds the migration modifier array described in class attributes

        Returns
        -------
        modifiers
        """
        
        migration_modifiers = np.zeros((len(self.migration_modifier_epochs), len(self.connections)), dtype="object")
        for j, modifier_string in enumerate(self.connections["migration_modifier"]):
            modifier_string = modifier_string.split(",")
            for e in range(len(modifier_string)):
                time, mod = modifier_string[e].split(":")
                time = float(time)
                if (len(modifier_string) > (e+1)):
                    next_time = float(modifier_string[e+1].split(":")[0])
                    relevant_epochs = np.where((self.migration_modifier_epochs>=time)&(self.migration_modifier_epochs<next_time))[0]
                else:
                    relevant_epochs = np.where(self.migration_modifier_epochs>=time)[0]
                migration_modifiers[relevant_epochs, j] = mod
        return migration_modifiers

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
    
    def linearly_interpolate_suitabilties(self, timesteps):
        """Adds extra timepoints between epochs to smooth the transition between suitability values

        Parameters
        ----------
        timesteps: int
        """
        
        if (timesteps < 0) or (not isinstance(timesteps, int)):
            raise ValueError("`timesteps` must be a natural number (non-negative integer).")
        
        if timesteps > 0:

            suitability_epochs = []
            suitabilities = []
            already_included = list(self.suitability_epochs)
            for i in range(len(self.suitability_epochs)-1):
                a = self.suitability_epochs[i]
                b = self.suitability_epochs[i+1]

                suitability_epochs.append(a)
                suitabilities.append(self.suitabilities[i])

                step = (b - a) / (timesteps + 1)
                for j in range(1, timesteps + 1):
                    virtual = round(a + step * j)
                    if (virtual not in already_included):
                        suitability_epochs.append(virtual)
                        suitabilities.append(self.suitabilities[i]*(((timesteps+1)-j)/(timesteps+1)) + self.suitabilities[i+1]*(j/(timesteps+1)))
                        already_included.append(virtual)
            suitability_epochs.append(self.suitability_epochs[-1])
            suitabilities.append(self.suitabilities[-1])

            self.suitability_epochs = np.array(suitability_epochs)
            self.suitabilities = np.array(suitabilities)

            self.epochs = np.concatenate((self.suitability_epochs, self.migration_modifier_epochs))
            self.epochs = np.unique(self.epochs)
            self.epoch_durations = np.diff(self.epochs, append=(self.epochs[-1]-1))
