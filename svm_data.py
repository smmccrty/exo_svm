import h5py
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple, Optional, Any


def reademceeh5(fn: str, nburn: int, thin: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read MCMC data from an HDF5 file with burn-in removal and thinning.
    
    Parameters
    ----------
    fn : str
        Filename of the HDF5 file containing MCMC data
    nburn : int
        Number of burn-in samples to remove from the beginning
    thin : int
        Thinning factor (take every nth sample)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        samples : MCMC samples array with shape (n_steps, n_walkers, n_params)
        lnprob : Log probability values with shape (n_steps, n_walkers)
        
    Notes
    -----
    The function expects the HDF5 file to have a 'mcmc' group containing
    the samples and log probability data.
    """
    # Open HDF5 file and extract important data
    hf = h5py.File(fn, 'r')
    grps = [item for item in hf['mcmc'].values()]

    # Extract samples chain and log-likelihood
    samples = grps[1]
    lnprob = grps[2]

    # Remove burn-in and apply thinning
    samples = samples[nburn:-1, :, :]
    samples = samples[0::thin, :, :]
    lnprob = lnprob[nburn:-1, :]
    lnprob = lnprob[0::thin, :]

    # Close HDF5 file
    hf.close()

    return samples, lnprob


def sample_h5(fnr: str, nburn: int, thin: int, nsamps: int, 
              columns: List[int], ptype: str = 'vmr') -> np.ndarray:
    """
    Sample random data from rfast HDF5 MCMC files.
    
    Parameters
    ----------
    fnr : str
        Filename (with or without .h5 extension)
    nburn : int
        Number of burn-in samples to remove
    thin : int
        Thinning factor
    nsamps : int
        Number of random samples to extract
    columns : List[int]
        Column indices to extract from the samples
    ptype : str, optional
        Parameter type, by default 'vmr'
        Options: 'vmr' (volume mixing ratio), 'log_vmr', 'pp' (partial pressure), 'log_pp'
        
    Returns
    -------
    np.ndarray
        Transformed feature array with shape (nsamps, n_features)
    """
    # Ensure filename has .h5 extension
    if fnr[-3:] != '.h5':
        fnr += '.h5'
    
    # Read MCMC samples
    samples, _ = reademceeh5(fnr, nburn, thin)
    
    # Flatten walkers (combine steps and walkers dimensions)
    samples = samples.reshape(samples.shape[0] * samples.shape[1], samples.shape[2])

    # Extract random samples
    rsamples = samples[np.random.randint(len(samples), size=nsamps)]

    # Extract specified columns
    features = []
    for column in columns:
        features.append(rsamples[:, column])
    features = np.array(features)

    # Apply transformations based on ptype
    if ptype == 'vmr':
        pass  # No transformation needed
    elif ptype == 'log_vmr':
        features = 10**features
    elif ptype == 'pp':
        total_p = np.sum(features, axis=0, keepdims=True)
        features = features / total_p
    elif ptype == 'log_pp':
        features = 10**features
        total_p = np.sum(features, axis=0, keepdims=True)
        features = features / total_p
    else:
        raise ValueError(f"Unknown ptype: {ptype}. Must be one of 'vmr', 'log_vmr', 'pp', or 'log_pp'.")

    return features.transpose()


def parse_rfast_rpars_file(fname: str) -> List[str]:
    """
    Parse an RFAST parameters file to extract retrieved parameter names.
    
    Parameters
    ----------
    fname : str
        Path to the RFAST parameters file
        
    Returns
    -------
    List[str]
        List of parameter names that are marked as retrieved (status='y')
        
    Notes
    -----
    The function expects a pipe-delimited file with at least 3 columns:
    parameter_name | ... | retrieved_status
    
    Parameters starting with 'f' have the 'f' prefix removed and are
    converted to lowercase for standardization.
    """
    retrieved_params = []
 
    with open(fname, 'r') as file:
        lines = file.readlines()
    
    # Skip header line and process data lines
    for line in lines[1:]:
        line = line.strip()
        
        # Skip empty lines and comment lines
        if not line or line.startswith('#'):
            continue
            
        # Split by '|' and clean up whitespace
        parts = [part.strip() for part in line.split('|')]
        
        if len(parts) >= 3:  # Ensure we have enough columns
            param_name = parts[0]
            retrieved_status = parts[2]
            
            if retrieved_status == 'y':
                if param_name.startswith('f'):
                    # Remove the leading 'f' and standardize with lowercase
                    retrieved_params.append(param_name[1:].lower())
                else:
                    retrieved_params.append(param_name.lower())

    return retrieved_params


def extract_rfast_data_from_file(fname: str, nsamps: int, species: List[str], 
                                mc_params: Dict[str, int], rpars_fname: Optional[str] = None, 
                                rfast_columns: Optional[List[int]] = None, 
                                fill_species: Optional[str] = None, 
                                ptype: str = 'vmr') -> np.ndarray:
    """
    Extract RFAST data from HDF5 files with comprehensive parameter handling.
    
    Parameters
    ----------
    fname : str
        Path to the HDF5 data file
    nsamps : int
        Number of samples to extract
    species : List[str]
        List of species names to extract
    mc_params : Dict[str, int]
        Monte Carlo parameters containing 'nburn', 'thin', 'nstep', 'nwalkers'
    rpars_fname : Optional[str], optional
        Path to RFAST parameters file, by default None
    rfast_columns : Optional[List[int]], optional
        Direct column indices to use, by default None
    fill_species : Optional[str], optional
        Species to use as background/fill gas, by default None
    ptype : str, optional
        Parameter type, by default 'vmr'
        
    Returns
    -------
    np.ndarray
        Extracted feature array with shape (nsamps, n_species)
        
    Raises
    ------
    ValueError
        If requested samples exceed available samples, if species not found,
        if fill_species not in species list, or if neither rpars_fname nor
        rfast_columns is provided
        
    Notes
    -----
    Either rpars_fname or rfast_columns must be provided to specify which
    columns to extract. The fill_species parameter ensures that sum of all the 
    parameters is a physical value (e.g. the vmrs should add to 1).
    """
    # Standardize species names to lowercase
    species = [sp.lower() for sp in species]
    fill_species = fill_species.lower() if fill_species is not None else None

    # Extract Monte Carlo parameters
    burn = mc_params['nburn']
    thin = mc_params['thin']
    nstep = mc_params['nstep']
    nwalkers = mc_params['nwalkers']
    total_samples = nstep * nwalkers - burn
    
    # Validate sample count
    if nsamps > total_samples:
        raise ValueError(f"Requested number of samples ({nsamps}) exceeds total samples available ({total_samples}).")
    
    # Determine column indices to extract
    if rpars_fname is not None:
        rfast_params = parse_rfast_rpars_file(rpars_fname)
        species_idx = []
        for sp in species:
            if sp in rfast_params:
                species_idx.append(rfast_params.index(sp))
            else:
                raise ValueError(f"Species '{sp}' not found in the Rfast params list.")
    elif rfast_columns is not None:
        species_idx = rfast_columns
    else:
        raise ValueError("Either rpars_fname or rfast_columns must be provided.")
        
    # Read the HDF5 file and extract features
    features = sample_h5(fname, burn, thin, nsamps, species_idx, ptype=ptype)

    # Fill if fill_species is specified
    if fill_species is not None:
        if fill_species not in species:
            raise ValueError(f"Fill set to bg: {fill_species}, but this is not one of the species: {species}")
        fill_idx = species.index(fill_species)
        features[:, fill_idx] += 1 - np.sum(features, axis=1)

    return features


def load_training_data(fname: str, species: List[str], header_row: Optional[int] = None, 
                      minimum_value: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training data from a CSV file for machine learning applications.
    
    Parameters
    ----------
    fname : str
        Path to the CSV file containing training data
    species : List[str]
        List of species names (columns will be prefixed with 'f')
    header_row : Optional[int], optional
        Row number to use as header, by default None
    minimum_value : Optional[float], optional
        Minimum value to clip data to, by default None
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        data : Feature array with shape (n_samples, n_species)
        labels : Boolean array indicating biotic (True) or abiotic (False)
        refs : Array of reference/notes strings
        
    Notes
    -----
    The function expects the CSV to have:
    - A 'Biotic (Y/N)' column for labels
    - Species columns with 'f' prefix (e.g., 'fh2o', 'fco2')
    - A 'Reference/Notes' column
    
    Rows with NaN values are automatically removed.
    """
    # Load CSV data
    df = pd.read_csv(fname, header=header_row)
    
    # Extract and convert labels
    labels = df['Biotic (Y/N)'].to_numpy()
    labels = np.array([1 if label == 'Y' else 0 for label in labels]).astype(bool)
    
    # Extract species data (with 'f' prefix)
    csv_species = [f'f{spec}' for spec in species]
    data = np.array([df[spec].to_numpy() for spec in csv_species])
    
    # Apply minimum value clipping if specified
    if minimum_value is not None:
        data = np.clip(data, a_min=minimum_value, a_max=None)
    
    # Remove NaN rows
    mask = ~np.isnan(data[0])
    data = data[:, mask]
    data = np.array(data.transpose())
    labels = labels[mask]
    
    # Extract references
    refs = df['Reference/Notes'].to_numpy()
    
    return data, labels, refs


class SVMDatasetManager:
    """
    Manages multiple datasets with their associated parameters and file configurations.
    
    Attributes
    ----------
    datasets : Dict[str, Dict[str, Any]]
        Dictionary storing loaded datasets with their data and metadata
    dataset_configs : Dict[str, Dict[str, Any]]
        Dictionary storing dataset configurations for reloading
        
    Examples
    --------
    >>> manager = SVMDatasetManager()
    >>> file_config = {
    ...     'data_files': ['snr40_2um.h5', 'snr20_2um.h5'],
    ...     'snr': [40, 20],
    ...     'wavelength': ['2um', '2um'],
    ...     'rpars_files': ['rfast_rpars.txt', 'rfast_rpars.txt']
    ... }
    >>> mc_params = {'nburn': 1000, 'thin': 10, 'nstep': 5000, 'nwalkers': 50}
    >>> manager.add_dataset('test_data', '/path/to/data', file_config, 
    ...                     1000, ['h2o', 'co2'], mc_params)
    """
    
    def __init__(self):
        """Initialize the SVMDatasetManager with empty dataset storage."""
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.dataset_configs: Dict[str, Dict[str, Any]] = {}
    
    def add_dataset(self, name: str, data_dir: str, file_config: Dict[str, List], 
                   nsamps: int, species: List[str], mc_params: Dict[str, int], 
                   **extract_kwargs) -> None:
        """
        Add a dataset with its configuration.
        
        Parameters
        ----------
        name : str
            Name identifier for the dataset
        data_dir : str
            Directory containing the data files
        file_config : Dict[str, List]
            Dictionary where keys are parameter/file characteristics and values are lists.
            Must include 'data_files' and either 'rpars_files' or 'rfast_columns' keys.
            
            Example:
            {
                'data_files': ['snr40_2um.h5', 'snr20_2um.h5', ...],
                'snr': [40, 20, 10, 40, 20, 10],
                'wavelength': ['2um', '2um', '2um', '1um', '1um', '1um'],
                'rpars_files': ['rfast_rpars.txt', 'rfast_rpars.txt', ...],
                # OR
                'rfast_columns': [[0,1,2], [1,0,2], ...]
            }
        nsamps : int
            Number of samples to extract
        species : List[str]
            List of species names
        mc_params : Dict[str, int]
            Monte Carlo parameters containing 'nburn', 'thin', 'nstep', 'nwalkers'
        **extract_kwargs
            Additional keyword arguments for extract_rfast_data_from_file function.
            Can include: fill_species, ptype
            
        Raises
        ------
        ValueError
            If file_config is missing required keys or if lists have different lengths
        """
        # Validate file_config structure
        required_keys = ['data_files']
        optional_keys = ['rpars_files', 'rfast_columns']
        
        if 'data_files' not in file_config:
            raise ValueError("file_config must contain 'data_files' key")
            
        if not any(key in file_config for key in optional_keys):
            raise ValueError("file_config must contain either 'rpars_files' or 'rfast_columns' key")
        
        # Check all lists have the same length
        list_lengths = [len(v) for v in file_config.values()]
        if len(set(list_lengths)) > 1:
            raise ValueError("All lists in file_config must have the same length")
        
        # Store configuration
        self.dataset_configs[name] = {
            'data_dir': data_dir,
            'file_config': file_config,
            'nsamps': nsamps,
            'species': species,
            'mc_params': mc_params,
            'extract_kwargs': extract_kwargs
        }
        
        # Load the actual data
        self._load_dataset(name)
    
    def _load_dataset(self, name: str) -> None:
        """
        Load data for a specific dataset.
        
        Parameters
        ----------
        name : str
            Name of the dataset to load
            
        Notes
        -----
        This is an internal method that handles the actual data loading process
        based on the stored configuration.
        """
        config = self.dataset_configs[name]
        file_config = config['file_config']
        data_list = []
        
        num_files = len(file_config['data_files'])
        
        # Process each file in the configuration
        for i in range(num_files):
            data_file = os.path.join(config['data_dir'], file_config['data_files'][i])
            
            if 'rpars_files' in file_config:
                # Use parameter file approach
                rpars_file = os.path.join(config['data_dir'], file_config['rpars_files'][i])
                data = extract_rfast_data_from_file(
                    data_file, config['nsamps'], config['species'], 
                    config['mc_params'], rpars_fname=rpars_file,
                    **config['extract_kwargs']
                )
            else:
                # Use direct column specification approach
                rfast_column = file_config['rfast_columns'][i]
                data = extract_rfast_data_from_file(
                    data_file, config['nsamps'], config['species'], 
                    config['mc_params'], rfast_columns=rfast_column,
                    **config['extract_kwargs']
                )
            
            data_list.append(data)
        
        # Create parameter list from file_config (excluding file names)
        param_keys = [k for k in file_config.keys() 
                     if k not in ['data_files', 'rpars_files', 'rfast_columns']]
        params = []
        for i in range(num_files):
            param_dict = {key: file_config[key][i] for key in param_keys}
            params.append(param_dict)
        
        # Store the loaded dataset
        self.datasets[name] = {
            'data': np.array(data_list),
            'params': params,
            'file_config': file_config
        }
    
    def get_dataset(self, name: str) -> Dict[str, Any]:
        """
        Get dataset by name.
        
        Parameters
        ----------
        name : str
            Name of the dataset to retrieve
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'data', 'params', and 'file_config' keys
            
        Raises
        ------
        ValueError
            If dataset name is not found
        """
        if name not in self.datasets:
            available = list(self.datasets.keys())
            raise ValueError(f"Dataset '{name}' not found. Available: {available}")
        return self.datasets[name]
    
    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns
        -------
        List[str]
            List of dataset names
        """
        return list(self.datasets.keys())
    
    def get_param_values(self, name: str, param_name: str) -> List[Any]:
        """
        Get unique values for a specific parameter across a dataset.
        
        Parameters
        ----------
        name : str
            Name of the dataset
        param_name : str
            Name of the parameter to query
            
        Returns
        -------
        List[Any]
            Sorted list of unique parameter values
            
        Raises
        ------
        ValueError
            If dataset or parameter name is not found
        """
        dataset = self.get_dataset(name)
        file_config = dataset['file_config']
        
        if param_name not in file_config:
            available_params = [k for k in file_config.keys() 
                              if k not in ['data_files', 'rpars_files', 'rfast_columns']]
            raise ValueError(f"Parameter '{param_name}' not found in dataset '{name}'. "
                           f"Available parameters: {available_params}")
        
        return sorted(list(set(file_config[param_name])))