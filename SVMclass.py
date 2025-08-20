from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from svm_data import SVMDatasetManager
import seaborn as sns
from collections import defaultdict
from matplotlib.ticker import ScalarFormatter

class SVMclass:
    def __init__(self, kernel='rbf', class_weight='balanced', C=1, gamma='scale', 
                 random_state=42, use_ensemble=True, n_ensemble_models=100, log_scale=False, cache_size=1000):
        """
        Main SVM class for classifying exoplanets.
        
        Parameters:
        -----------
        kernel : str, default='rbf'
            SVM kernel type
        class_weight : str or dict, default='balanced'
            Class weight strategy
        C : float, default=1
            Regularization parameter
        gamma : str or float, default='scale'
            Kernel coefficient
        random_state : int, default=42
            Random state for reproducibility
        use_ensemble : bool, default=True
            Whether to train multiple SVM models for ensemble predictions
        n_ensemble_models : int, default=100
            Number of ensemble models to train (only used if use_ensemble=True)
        log_scale : bool, default=False
            Whether to apply log scaling to features before standardization
        """
        self.log_scale = log_scale
        self.use_ensemble = use_ensemble
        self.n_ensemble_models = n_ensemble_models
        self.scalers = []
        self.models = []
        self.kernel = kernel
        self.class_weight = class_weight
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.cache_size = cache_size
        
        # Main scaler and model (always created)
        self.scaler = StandardScaler()
        self.clf = None

        # Add dataset management
        self.dataset_manager = SVMDatasetManager()
        self.prediction_results = {}

    def train(self, X, y, test_size=0.2, n_bootstrap=100):
        """
        Train SVM model(s) - either single model or ensemble based on use_ensemble parameter.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        test_size : float, default=0.5
            Proportion of data to use for testing
        n_bootstrap : int, default=100
            Number of bootstrap samples when calculating model performance on training/test data.
        """
        # Apply log scaling if requested
        if self.log_scale:
            if not np.all(X > 0):
                raise ValueError("Log scaling requires all feature values to be positive.")
            X = np.log10(X)
            
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state)
        
        # Always create main scaler and model
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.clf = svm.SVC(kernel=self.kernel, class_weight=self.class_weight, 
                          C=self.C, gamma=self.gamma, 
                          random_state=self.random_state, probability=True, cache_size=self.cache_size)
        self.clf.fit(self.X_train_scaled, self.y_train)
        
        if self.use_ensemble:
            # Train ensemble models on bootstrap samples
            self.scalers = []
            self.models = []
            
            np.random.seed(self.random_state)
            for i in tqdm(range(self.n_ensemble_models), total=self.n_ensemble_models, desc="Training ensemble models"):
                # Create bootstrap sample with replacement
                n_samples = len(self.X_train)
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = self.X_train[bootstrap_indices]
                y_bootstrap = self.y_train[bootstrap_indices]
                
                # Create scaler and model for this bootstrap
                scaler = StandardScaler()
                model = svm.SVC(kernel=self.kernel, class_weight=self.class_weight, 
                              C=self.C, gamma=self.gamma, 
                              random_state=self.random_state + i, probability=True)
                
                # Fit scaler and model
                X_bootstrap_scaled = scaler.fit_transform(X_bootstrap)
                model.fit(X_bootstrap_scaled, y_bootstrap)
                
                # Store scaler and model
                self.scalers.append(scaler)
                self.models.append(model)
        else:
            # Use single model as "ensemble" for consistent API
            self.scalers = [self.scaler]
            self.models = [self.clf]

        print("Predicting on training and test sets...")
        
        # Get predictions for train and test sets
        _, self.y_train_probs = self.predict_proba_ensemble(self.X_train, n_bootstrap=100, log=False)
        self.y_train_pred = np.array([prob['median'] for prob in self.y_train_probs]) > 0.5
        
        _, self.y_test_probs = self.predict_proba_ensemble(self.X_test, n_bootstrap=100, log=False)
        self.y_test_pred = np.array([prob['median'] for prob in self.y_test_probs]) > 0.5

    def predict(self, X):
        """Predict using a single SVM. Returns value in range (-1,1) (not calibrated as a probability)."""
        if self.log_scale:
            if not np.all(X > 0):
                raise ValueError("Log scaling requires all feature values to be positive.")
            X = np.log10(X)

        X_scaled = self.scaler.transform(X)
        return X_scaled, self.clf.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict using a single SVM. Returns value in range (0,1) (calibrated as a probability)."""
        if self.log_scale:
            if not np.all(X > 0):
                raise ValueError("Log scaling requires all feature values to be positive.")
            X = np.log10(X)
        X_scaled = self.scaler.transform(X)
        return X_scaled, self.clf.predict_proba(X_scaled)[:,1]

    def _get_all_model_predictions(self, X, log=None):
        """
        Helper function to batch compute all model predictions once.
        Returns predictions array of shape (n_models, n_samples).
        """
        # Apply log scaling if requested
        if log is None:
            if self.log_scale:
                if not np.all(X > 0):
                    raise ValueError("Log scaling requires all feature values to be positive.")
                X_processed = np.log10(X)
            else:
                X_processed = X
        elif log:
            X_processed = np.log10(X)
        else:
            X_processed = X
        
        # Pre-allocate predictions array
        n_samples = len(X)
        n_models = len(self.models)
        all_predictions = np.empty((n_models, n_samples))
        
        # Batch compute all model predictions
        for i, (scaler, model) in enumerate(zip(self.scalers, self.models)):
            X_scaled = scaler.transform(X_processed)
            all_predictions[i] = model.predict_proba(X_scaled)[:, 1]
        
        return X_processed, all_predictions

    def predict_proba_ensemble(self, X, n_bootstrap=100, confidence_level=0.95, log=None):
        """
        Predict probabilities using ensemble of models with bootstrap uncertainty quantification.
        For each data point, bootstrap samples from the ensemble model predictions to get
        confidence intervals around the median prediction.
        
        Parameters:
        -----------
        X : array-like
            Input features
        n_bootstrap : int, default=100
            Number of bootstrap samples for confidence interval estimation
        confidence_level : float, default=0.95
            Confidence level for intervals (0.95 = 95% CI)
        log: bool, optional
            If True, apply log scaling to input features before processing. 
            If none, use self.log_scale to decide.
            
        Returns:
        --------
        X_processed : array
            Processed input features (log-scaled if requested)
        prediction_results : list of dict
            For each data point, a dictionary containing:
            - 'median': float, median prediction across ensemble models
            - 'ci_lower': float, lower bound of confidence interval
            - 'ci_upper': float, upper bound of confidence interval
            - 'ensemble_size': int, number of models used
        """
        # Get all model predictions in batch
        X_processed, all_predictions = self._get_all_model_predictions(X, log=log)
        n_models, n_samples = all_predictions.shape
        
        if n_models == 1:
            # Single model case - no bootstrap uncertainty needed
            predictions = all_predictions[0]
            return X_processed, [
                {'median': pred, 'ci_lower': pred, 'ci_upper': pred}
                for pred in predictions
            ]
        
        # Vectorized bootstrap sampling
        # Generate all bootstrap indices at once: (n_samples, n_bootstrap, n_models)
        np.random.seed(self.random_state)
        bootstrap_indices = np.random.randint(0, n_models, size=(n_samples, n_bootstrap, n_models))
        
        # Use advanced indexing to get bootstrap samples: (n_samples, n_bootstrap, n_models)
        bootstrap_samples = all_predictions[bootstrap_indices, np.arange(n_samples)[:, None, None]]
        
        # Compute bootstrap medians for all samples at once: (n_samples, n_bootstrap)
        bootstrap_medians = np.median(bootstrap_samples, axis=2)
        
        # Compute actual medians for each sample: (n_samples,)
        actual_medians = np.median(all_predictions, axis=0)
        
        # Compute confidence intervals for all samples at once
        alpha = 1 - confidence_level
        percentiles = [100 * alpha/2, 100 * (1 - alpha/2)]
        ci_bounds = np.percentile(bootstrap_medians, percentiles, axis=1)  # (2, n_samples)
        
        # Package results
        prediction_results = [
            {
                'median': actual_medians[i],
                'ci_lower': ci_bounds[0, i],
                'ci_upper': ci_bounds[1, i],
            }
            for i in range(n_samples)
        ]
        
        return X_processed, prediction_results

    def distribution_predict(self, X, n_bootstrap=100, confidence_level=0.95, log=None):
        """
        Predict using bootstrap sampling to separate model and distribution uncertainties.
        This method computes separate confidence intervals for model uncertainty 
        (across ensemble members) and distribution uncertainty (across input samples),
        then combines them in quadrature around the overall median.
        
        Parameters:
        -----------
        X : array-like
            Input features (multiple data points)
        n_bootstrap : int, default=100
            Number of bootstrap samples for confidence interval estimation
        confidence_level : float, default=0.95
            Confidence level for intervals (0.95 = 95% CI)
        log: bool, optional
            If True, apply log scaling to input features before processing. 
            If none, use self.log_scale to decide.
        
        Returns:
        --------
        X_processed : array
            Processed input features
        individual_probs : array
            Individual probability predictions with model uncertainty only
        distribution_stats : dict
            Statistics about the distribution including:
            - median: float, median of all predictions (fiducial prediction)
            - ci_lower: float, lower bound of combined confidence interval
            - ci_upper: float, upper bound of combined confidence interval
            - model_uncertainty_lower: float, model uncertainty in negative direction
            - model_uncertainty_upper: float, model uncertainty in positive direction
            - distribution_uncertainty_lower: float, distribution uncertainty in negative direction
            - distribution_uncertainty_upper: float, distribution uncertainty in positive direction
        """

        # Get all model predictions in batch
        X_processed, all_predictions = self._get_all_model_predictions(X, log=log)
        n_models, n_samples = all_predictions.shape
        
        # Calculate overall fiducial prediction once
        overall_median = np.median(all_predictions)
        
        # Single model case - early return
        if n_models == 1:
            individual_probs = [
                {'median': pred, 'ci_lower': pred, 'ci_upper': pred}
                for pred in all_predictions[0]
            ]
            return X_processed, individual_probs, individual_probs[0] if n_samples == 1 else individual_probs
        
        # Vectorized bootstrap operations
        np.random.seed(self.random_state)
        alpha = 1 - confidence_level
        percentiles = [100 * alpha/2, 100 * (1 - alpha/2)]
        
        # Model uncertainty: vectorized bootstrap across models for ALL samples at once
        model_bootstrap_indices = np.random.randint(0, n_models, size=(n_bootstrap, n_models))
        model_bootstrap_samples = all_predictions[model_bootstrap_indices]  # (n_bootstrap, n_models, n_samples)
        model_bootstrap_medians = np.median(model_bootstrap_samples, axis=1)  # (n_bootstrap, n_samples)
        
        # Individual sample predictions with model uncertainty
        sample_medians = np.median(all_predictions, axis=0)  # (n_samples,)
        individual_ci_bounds = np.percentile(model_bootstrap_medians, percentiles, axis=0)  # (2, n_samples)
        
        individual_probs = [
            {
                'median': sample_medians[i],
                'ci_lower': individual_ci_bounds[0, i],
                'ci_upper': individual_ci_bounds[1, i],
            }
            for i in range(n_samples)
        ]
        
        if n_samples == 1:
            return X_processed, individual_probs, individual_probs[0]
        
        # Model uncertainty: compute average uncertainties from vectorized results
        model_uncertainties = individual_ci_bounds - sample_medians[None, :]  # (2, n_samples)
        avg_model_uncertainty_lower = np.median(-model_uncertainties[0])  # Make positive
        avg_model_uncertainty_upper = np.median(model_uncertainties[1])
        
        # Distribution uncertainty: vectorized bootstrap across samples for ALL models
        np.random.seed(self.random_state + 1)
        dist_bootstrap_indices = np.random.randint(0, n_samples, size=(n_bootstrap, n_samples))
        dist_bootstrap_samples = all_predictions[:, dist_bootstrap_indices]  # (n_models, n_bootstrap, n_samples)
        dist_bootstrap_medians = np.median(dist_bootstrap_samples, axis=2)  # (n_models, n_bootstrap)
        
        # Calculate distribution uncertainty for all models at once
        model_medians = np.median(all_predictions, axis=1)  # (n_models,)
        dist_ci_bounds = np.percentile(dist_bootstrap_medians, percentiles, axis=1)  # (2, n_models)
        dist_uncertainties = dist_ci_bounds - model_medians[None, :]  # (2, n_models)
        
        avg_distribution_uncertainty_lower = np.median(-dist_uncertainties[0])  # Make positive
        avg_distribution_uncertainty_upper = np.median(dist_uncertainties[1])
        
        # Combine uncertainties
        total_uncertainty_lower = np.sqrt(avg_model_uncertainty_lower**2 + avg_distribution_uncertainty_lower**2)
        total_uncertainty_upper = np.sqrt(avg_model_uncertainty_upper**2 + avg_distribution_uncertainty_upper**2)
        
        distribution_stats = {
            'median': overall_median,
            'ci_lower': overall_median - total_uncertainty_lower,
            'ci_upper': overall_median + total_uncertainty_upper,
            'model_uncertainty_lower': avg_model_uncertainty_lower,
            'model_uncertainty_upper': avg_model_uncertainty_upper,
            'distribution_uncertainty_lower': avg_distribution_uncertainty_lower,
            'distribution_uncertainty_upper': avg_distribution_uncertainty_upper,
        }
        
        return X_processed, individual_probs, distribution_stats
    
    def add_dataset(self, name, data_dir, file_config, 
                   nsamps, species, mc_params, **extract_kwargs):
        """Add a dataset to the manager. See SVMDatasetManager for details."""
        self.dataset_manager.add_dataset(
            name, data_dir, file_config,
            nsamps, species, mc_params, **extract_kwargs
        )

    def predict_on_datasets(self, dataset_names=None, confidence_level=0.95, n_bootstrap=100):
        """
        Generate predictions for one or more datasets.
        
        Parameters:
        -----------
        dataset_names : list or str, optional
            Names of datasets to predict on. If None, predict on all datasets.
        confidence_level : float
            Confidence level for uncertainty quantification
        n_bootstrap : int
            Number of bootstrap samples for uncertainty
        """
        if dataset_names is None:
            dataset_names = self.dataset_manager.list_datasets()
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        for dataset_name in dataset_names:
            dataset = self.dataset_manager.get_dataset(dataset_name)
            data_array = dataset['data']
            
            predictions = []
            ci_lowers = []
            ci_uppers = []
            
            print(f"Generating predictions for dataset: {dataset_name}")
            for i, data in enumerate(tqdm(data_array, desc="Processing files")):
                _, _, result = self.distribution_predict(
                    data, n_bootstrap=n_bootstrap, confidence_level=confidence_level
                )
                predictions.append(result['median'])
                ci_lowers.append(result['ci_lower'])
                ci_uppers.append(result['ci_upper'])
            
            self.prediction_results[dataset_name] = {
                'predictions': np.array(predictions),
                'ci_lower': np.array(ci_lowers),
                'ci_upper': np.array(ci_uppers),
                'params': dataset['params']
            }




    ##### Plotting/analysis funcs ########



    def plot_species_distributions_multi(self, dataset_names, species, species_idx, 
                                       color_param=None, linestyle_param=None,
                                       training_data=None, training_truth=None, 
                                       title_prefix="Species Distributions",
                                       color_map=None, linestyle_map=None):
        """
        Plot species distributions for multiple datasets in a single figure.
        
        Parameters:
        -----------
        dataset_names : list
            List of dataset names to plot
        species : list
            List of species names
        species_idx : list
            Indices of species to plot
        color_param : str, optional
            Parameter name to use for coloring (e.g., 'snr')
        linestyle_param : str, optional
            Parameter name to use for linestyle variation (e.g., 'wavelength')
        training_data : array-like, optional
            Training data for comparison
        training_truth : array-like, optional
            Training labels for comparison
        title_prefix : str
            Prefix for plot titles
        color_map : dict, optional
            Custom mapping from parameter values to colors
        linestyle_map : dict, optional
            Custom mapping from parameter values to linestyles
        """
        n_datasets = len(dataset_names)
        n_species = len(species_idx)
        n_rows = n_datasets + (1 if training_data is not None else 0)
        
        fig, axes = plt.subplots(n_rows, n_species, figsize=(4*n_species, 4*n_rows))
        
        # Handle single species case
        if n_species == 1:
            axes = axes.reshape(-1, 1)
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for dataset_idx, dataset_name in enumerate(dataset_names):
            dataset = self.dataset_manager.get_dataset(dataset_name)
            data_array = dataset['data']
            file_config = dataset['file_config']
            
            # Get parameter values for coloring and linestyle
            color_values = file_config.get(color_param, ['unknown'] * len(file_config['data_files'])) if color_param else None
            linestyle_values = file_config.get(linestyle_param, ['unknown'] * len(file_config['data_files'])) if linestyle_param else None
            
            #row_offset = dataset_idx * (2 if training_data is not None else 1)
            
            for i, idx in enumerate(species_idx):
                # Plot dataset distributions
                self._plot_species_condition(
                    data_array, idx, axes[dataset_idx, i], 
                    f'{dataset_name} - {species[idx]}',
                    color_values, linestyle_values, color_param, linestyle_param,
                    color_map, linestyle_map
                )
                

        for i, idx in enumerate(species_idx):
            # Plot training data if provided
            if training_data is not None:
                self._plot_training_data(
                    training_data, training_truth, idx, 
                    axes[-1, i],
                    f'{species[idx]} (Training)'
                )
        
        plt.suptitle(f'{title_prefix}', fontsize=16)
        plt.tight_layout()
        plt.show()

    def _plot_species_condition(self, data, species_idx, ax, title, 
                                       color_values, linestyle_values, 
                                       color_param, linestyle_param,
                                       color_map=None, linestyle_map=None):
        """
        Internal function for plot_species_distributions_multi.
        
        Parameters:
        -----------
        data : array
            Data array of shape (n_files, n_samples, n_species)
        species_idx : int
            Index of species to plot
        ax : matplotlib axis
            Axis to plot on
        title : str
            Plot title
        color_values : list
            Values for color parameter (can be None)
        linestyle_values : list
            Values for linestyle parameter (can be None)
        color_param : str
            Name of color parameter
        linestyle_param : str
            Name of linestyle parameter
        color_map : dict, optional
            Custom color mapping
        linestyle_map : dict, optional
            Custom linestyle mapping
        """
        species_data = data[:, :, species_idx]
        log_species_data = np.log10(species_data)
        log_min = np.min(log_species_data)
        log_max = np.max(log_species_data)
        bins = np.linspace(log_min, log_max, 30)
        
        # Create default color and linestyle maps
        if color_values is not None:
            unique_colors = sorted(list(set(color_values)))
            if color_map is None:
                # Use default colormap
                if len(unique_colors) <= 10:
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_colors)))
                else:
                    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_colors)))
                color_map = dict(zip(unique_colors, colors))
        
        if linestyle_values is not None:
            unique_linestyles = sorted(list(set(linestyle_values)))
            if linestyle_map is None:
                # Use default linestyles
                default_linestyles = ['-', '--', '-.', ':']
                linestyle_map = {}
                for i, val in enumerate(unique_linestyles):
                    linestyle_map[val] = default_linestyles[i % len(default_linestyles)]
        
        for i in range(len(color_values) if color_values else len(log_species_data)):
            log_data_subset = log_species_data[i, :]
            
            # Determine color and linestyle
            if color_values is not None:
                color = color_map.get(color_values[i], 'black')
            else:
                color = plt.cm.tab10(i % 10)
            
            if linestyle_values is not None:
                linestyle = linestyle_map.get(linestyle_values[i], '-')
            else:
                linestyle = '-'
            
            # Create label
            label_parts = []
            if color_param and color_values:
                label_parts.append(f'{color_param}={color_values[i]}')
            if linestyle_param and linestyle_values:
                label_parts.append(f'{linestyle_param}={linestyle_values[i]}')
            
            if not label_parts:
                label = f'File {i}'
            else:
                label = ', '.join(label_parts)
            
            ax.hist(log_data_subset, bins=bins, alpha=0.6, 
                   color=color, linestyle=linestyle, histtype='step', 
                   linewidth=2, density=True, label=label)
        
        ax.set_xlabel('log10(Abundance)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(fontsize=8)#, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    def _plot_training_data(self, training_data, training_truth, species_idx, ax, title):
        """Plot training data distributions. Internal function for plot_species_distributions_multi"""
        log_training_species_data = np.log10(training_data[:, species_idx])
        log_min = np.min(log_training_species_data)
        log_max = np.max(log_training_species_data)
        bins = np.linspace(log_min, log_max, 30)
        
        log_true_samples = log_training_species_data[training_truth]
        log_false_samples = log_training_species_data[~training_truth]
        
        if len(log_true_samples) > 0:
            ax.hist(log_true_samples, bins=bins, alpha=0.7, 
                   color='green', linestyle='-', histtype='step', 
                   linewidth=2, density=True, label='Biotic')
        
        if len(log_false_samples) > 0:
            ax.hist(log_false_samples, bins=bins, alpha=0.7, 
                   color='red', linestyle='--', histtype='step', 
                   linewidth=2, density=True, label='Abiotic')
        
        ax.set_xlabel('log10(Abundance)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_predictions_vs_parameters(self, dataset_names=None, param_name='snr', 
                                     group_by=None, title="Predictions vs Parameters"):
        """
        Plot predictions against a specified parameter with uncertainty bands.
        
        Parameters:
        -----------
        dataset_names : list, optional
            Dataset names to include. If None, use all datasets with predictions.
        param_name : str
            Parameter name to plot against (e.g., 'snr', 'wavelength')
        group_by : str, optional
            Parameter to group by (e.g., if param_name='snr', group_by='wavelength')
        title : str
            Plot title
        """
        if dataset_names is None:
            dataset_names = [name for name in self.prediction_results.keys()]
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        n_datasets = len(dataset_names)
        fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
        if n_datasets == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16)
        
        for i, dataset_name in enumerate(dataset_names):
            if dataset_name not in self.prediction_results:
                print(f"No predictions found for dataset: {dataset_name}")
                continue
            
            results = self.prediction_results[dataset_name]
            predictions = results['predictions']
            ci_lower = results['ci_lower']
            ci_upper = results['ci_upper']
            params = results['params']
            
            ax = axes[i]
            
            if group_by is None:
                # Simple plot without grouping
                param_values = [p.get(param_name, 'unknown') for p in params]
                
                # Sort by parameter values for better plotting
                sorted_indices = np.argsort([float(v) if str(v).replace('.','').isdigit() else 0 
                                           for v in param_values])
                
                x_vals = np.array(param_values)[sorted_indices]
                y_vals = predictions[sorted_indices]
                y_lower = ci_lower[sorted_indices]
                y_upper = ci_upper[sorted_indices]
                
                yerr = np.row_stack((np.abs(y_lower - y_vals), y_upper - y_vals))
                ax.errorbar(x_vals, y_vals, yerr=yerr, 
                           fmt='o', capsize=5, capthick=2, markersize=8)
            else:
                # Group by another parameter
                groups = defaultdict(list)
                for j, p in enumerate(params):
                    group_key = p.get(group_by, 'unknown')
                    groups[group_key].append(j)
                
                for group_key, indices in groups.items():
                    param_values = [params[j].get(param_name, 'unknown') for j in indices]
                    
                    # Sort by parameter values
                    sorted_local_indices = np.argsort([float(v) if str(v).replace('.','').isdigit() else 0 
                                                     for v in param_values])
                    
                    x_vals = np.array(param_values)[sorted_local_indices]
                    y_vals = predictions[np.array(indices)[sorted_local_indices]]
                    y_lower = ci_lower[np.array(indices)[sorted_local_indices]]
                    y_upper = ci_upper[np.array(indices)[sorted_local_indices]]
                    
                    yerr = np.row_stack((np.abs(y_lower - y_vals), y_upper - y_vals))
                    ax.errorbar(x_vals, y_vals, yerr=yerr, 
                               fmt='o', capsize=5, capthick=2, markersize=8, 
                               label=f'{group_by}={group_key}')
                
                ax.legend()
            
            ax.set_xlabel(param_name.upper())
            ax.set_ylabel('Prediction')
            ax.set_title(f'{dataset_name}')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true=None, y_pred=None):
        """Plot confusion matrix using seaborn. 
        If no data is provided the test portion of the training set will be used."""

        # If no data provided, use training data from last train() call
        if y_true is None or y_pred is None:
            y_pred = self.y_test_pred
            y_true = self.y_test

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def plot_svm_pairwise(self, X=None, y=None, y_pred_proba=None, feature_names=None, point_size=50, scale='log'):
        """
        Plot pairwise feature comparisons showing prediction probabilities.
        
        Parameters:
        -----------
        If X,y,y_pred_proba are not provided, use the test data from the last call to train().

        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            True target values (binary: 0 or 1)
        y_pred_proba : array-like of shape (n_samples,) or array of dicts of length (n_samples) with medians
            Predicted probabilities from SVM model (continuous values between 0 and 1)
        feature_names : list, optional
            Names of features for axis labels
        point_size : int, optional (default=50)
            Size of scatter plot points
        scale : str, optional (default='log')
            Scale for the axes ('log', 'linear', etc.)
        """

        # If no data provided, use training data from last train() call
        if X is None or y is None or y_pred_proba is None:
            if not hasattr(self, 'X_test') or not hasattr(self, 'y_test') or not hasattr(self, 'y_test_probs'):
                raise ValueError("No training data available. Please train the model first.")
            X_plot = self.X_test
            y = self.y_test
            y_pred_proba = self.y_test_probs
        else:
            # Apply log scaling if it was used during training
            if self.log_scale:
                if not np.all(X > 0):
                    raise ValueError("Log scaling requires all feature values to be positive.")
                X_plot = np.log10(X)
            else:
                X_plot = X

        if isinstance(y_pred_proba[0], dict):
            # If y_pred_proba is a dict, extract the median probabilities
            y_pred_proba = np.array([prob['median'] for prob in y_pred_proba])
        
        # Get number of features
        n_features = X_plot.shape[1]
        
        # Use default feature names if none provided
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        # Get all pairwise combinations of features
        feature_pairs = list(combinations(range(n_features), 2))
        n_pairs = len(feature_pairs)
        
        # Calculate grid dimensions for subplots
        n_cols = int(np.ceil(np.sqrt(n_pairs)))
        n_rows = int(np.ceil(n_pairs / n_cols))
        
        # Create figure with significantly more height for spacing
        fig = plt.figure(figsize=(4*n_cols, 4*n_rows + 1.2))
        
        # Create a colorbar axes at the top with more space
        cbar_height = 0.02
        cbar_top_position = 0.98  # Position even higher
        
        # Positions for the two colorbars (left half and right half of figure)
        cbar_pos_left = plt.axes([0.125, cbar_top_position, 0.35, cbar_height])
        cbar_pos_right = plt.axes([0.525, cbar_top_position, 0.35, cbar_height])
        
        # Create colormaps for positive and negative cases
        cmap_pos = plt.cm.Blues
        cmap_neg = plt.cm.Reds
        
        # Create dummy scatter plots for the colorbar
        sm_pos = plt.cm.ScalarMappable(cmap=cmap_pos, norm=plt.Normalize(0, 1))
        sm_neg = plt.cm.ScalarMappable(cmap=cmap_neg, norm=plt.Normalize(0, 1))
        
        # Add separate colorbars for each class with clear titles
        cbar_pos = fig.colorbar(sm_pos, cax=cbar_pos_left, orientation='horizontal')
        cbar_pos_left.text(0.5, 1.5, 'True Class = 1 (probability)', 
                        ha='center', va='bottom', transform=cbar_pos_left.transAxes)
        
        cbar_neg = fig.colorbar(sm_neg, cax=cbar_pos_right, orientation='horizontal')
        cbar_pos_right.text(0.5, 1.5, 'True Class = 0 (probability)', 
                        ha='center', va='bottom', transform=cbar_pos_right.transAxes)
        
        # Create subplot grid with more space between plots
        gs = plt.GridSpec(n_rows, n_cols, left=0.1, right=0.9, 
                        bottom=0.1, top=0.92,  # More space at top
                        wspace=0.4, hspace=0.3)  # More space between plots
        
        # Plot each feature pair
        for idx, (i, j) in enumerate(feature_pairs):
            ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
            
            # Create masks for true class
            pos_mask = (y == 1)
            neg_mask = (y == 0)
            
            # Plot points with color based on prediction probability
            if np.sum(pos_mask) > 0:
                scatter_pos = ax.scatter(X_plot[pos_mask, i], X_plot[pos_mask, j], 
                            c=y_pred_proba[pos_mask], cmap=cmap_pos, 
                            alpha=0.7, s=point_size, vmin=0, vmax=1)
            
            if np.sum(neg_mask) > 0:
                scatter_neg = ax.scatter(X_plot[neg_mask, i], X_plot[neg_mask, j], 
                            c=y_pred_proba[neg_mask], cmap=cmap_neg, 
                            alpha=0.7, s=point_size, vmin=0, vmax=1)
            
            # Set labels and title with adjusted positions
            ax.set_xlabel(feature_names[i], labelpad=10)  # Add padding to x-label
            ax.set_ylabel(feature_names[j], labelpad=10)  # Add padding to y-label
            ax.set_title(f'{feature_names[i]} vs {feature_names[j]}', pad=10)  # Add padding to title
            ax.set_yscale(scale)
            ax.set_xscale(scale)
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjusted rect to leave more room at top
        plt.show()
        return

    def detailed_performance_metrics(self, y_true=None, y_pred=None):
        """Calculate and print detailed performance metrics. 
        If no data is provided the test portion of the training set will be used."""        

        if y_true is None or y_pred is None:
            print("Using test data...")
            y_true = self.y_test
            y_pred = self.y_test_pred

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate specific metrics
        tn, fp, fn, tp = cm.ravel()
        
        # True Positive Rate (Sensitivity, Recall)
        tpr = tp / (tp + fn)
        # True Negative Rate (Specificity)
        tnr = tn / (tn + fp)
        # False Positive Rate
        fpr = fp / (fp + tn)
        # False Negative Rate
        fnr = fn / (fn + tp)
        # Precision
        precision = tp / (tp + fp)
        # F1 Score
        f1 = 2 * (precision * tpr) / (precision + tpr)
        
        print("Detailed Performance Metrics:")
        print("-" * 30)
        print(f"True Positive Rate (Sensitivity): {tpr:.3f}")
        print(f"True Negative Rate (Specificity): {tnr:.3f}")
        print(f"False Positive Rate: {fpr:.3f}")
        print(f"False Negative Rate: {fnr:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"F1 Score: {f1:.3f}")