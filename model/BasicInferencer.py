import torch
import math
import os
import time
import copy
import numpy as np
from tqdm import tqdm
from lib.logger import get_logger
from lib.metrics import All_Metrics
from lib.dataloader import get_feature
from lib.TrainInits import print_model_parameters


class Inferencer(object):
    """
    Inferencer for multi-stage spatial-temporal prediction model.
    
    This class performs inference using the complete trained pipeline:
    1. ADP (Adaptation): Predict unsensed locations from sensed history
    2. Forecast: Predict future values for sensed locations from all history data
    3. Aggregation: Combine historical and future information for final prediction
    
    It evaluates each stage independently and provides comprehensive metrics.
    """
    
    def __init__(self, sensed_locations, unsensed_locations, adp_model, forecast_model, agg_model, 
                 test_loader, scaler, args, device):
        """
        Initialize the multi-stage inferencer.
        
        Args:
            sensed_locations: Indices of locations with sensors
            unsensed_locations: Indices of locations without sensors
            adp_model: Trained adaptation model for unsensed location prediction
            forecast_model: Trained forecasting model for future prediction
            agg_model: Trained aggregation model for final prediction
            test_loader: Testing data loader
            scaler: Data scaler for normalization
            args: Inference arguments and hyperparameters
            device: Computing device (CPU/GPU)
        """
        super(Inferencer, self).__init__()
        
        # Models
        self.adp_model = adp_model
        self.forecast_model = forecast_model
        self.agg_model = agg_model
        
        # Location information
        self.sensed_locations = sensed_locations
        self.unsensed_locations = unsensed_locations
        
        # Data handling
        self.test_loader = test_loader
        self.scaler = scaler
        
        # Configuration
        self.args = args
        self.device = device
        
        # Ensure device is properly formatted for torch operations
        if isinstance(device, int):
            self.device_str = f'cuda:{device}'
        elif isinstance(device, torch.device):
            self.device_str = str(device)
        else:
            self.device_str = device
        
        # Path configuration
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        
        # Setup logging
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, debug=False)
        self.logger.info(f'Inference log path: {args.log_dir}')

    def _log_stage_results(self, stage_name, pred, true):
        """
        Log detailed results for a specific stage.
        
        Args:
            stage_name: Name of the stage ('Adaptation', 'Forecast', or 'Aggregation')
            pred: Predictions tensor
            true: Ground truth tensor
        """
        self.logger.info(f'********************{stage_name} Step********************')
        
        # Log metrics for each time horizon
        for t in range(true.shape[1]):
            mae, rmse, mape, _ = All_Metrics(pred[:, t, ...], true[:, t, ...],
                                           self.args.mae_thresh, self.args.mape_thresh)
            self.logger.info(f"Horizon {t + 1:02d}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape * 100:.4f}%")
        
        # Log overall average metrics
        mae, rmse, mape, _ = All_Metrics(pred, true, self.args.mae_thresh, self.args.mape_thresh)
        self.logger.info(f"Average Horizon, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape * 100:.4f}%")
    
    def _save_predictions(self, target_true, target_pred):
        """
        Save prediction results to disk for further analysis.
        
        Args:
            target_true: Ground truth target values
            target_pred: Predicted target values
        """
        target_true_path = os.path.join(self.args.log_dir, 'target_true.pth')
        target_pred_path = os.path.join(self.args.log_dir, 'target_pred.pth')
        
        torch.save(target_true, target_true_path)
        torch.save(target_pred, target_pred_path)
        
        self.logger.info(f'Predictions saved to: {target_pred_path}')
        self.logger.info(f'Ground truth saved to: {target_true_path}')

    def inference(self):
        """
        Perform comprehensive inference using the complete multi-stage pipeline.
        
        This method:
        1. Runs the full ADP -> Forecast -> Aggregation pipeline
        2. Evaluates each stage independently with detailed metrics
        3. Saves predictions for further analysis
        4. Provides comprehensive performance evaluation
        """
        # Set all models to evaluation mode
        self.adp_model.eval()
        self.forecast_model.eval()
        self.agg_model.eval()
        
        # Initialize result collectors for each stage
        history_unsensed_pred = []  # ADP stage predictions
        history_unsensed_true = []  # ADP stage ground truth
        future_sensed_pred = []     # Forecast stage predictions
        future_sensed_true = []     # Forecast stage ground truth
        target_pred = []            # Final aggregation predictions
        target_true = []            # Final target ground truth
        
        self.logger.info('Starting inference on test dataset...')
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Inference Progress"):
                X, unsensed_X, sensed_Y, target = batch
                # Data shape descriptions:
                # X shape [bs, T, M, C] batchsize, T=12, M= # of sensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
                # unsensed_X shape [bs, T, M', C] batchsize, T=12, M'= # of unsensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
                # sensed_Y shape [bs, T', M, C] batchsize, T'=96, M= # of sensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
                # target shape [bs, T', M', C] batchsize, T'=96, M'= # of unsensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
                
                label = target[..., :self.args.output_dim]
                
                # Stage 1: Adaptation - predict unsensed locations from sensed history
                unsensed_output = self.adp_model(X[..., :-1])  # output shape [bs, T, M', 1]
                
                # Generate output's other features: time-of-day, day-of-week, rank, time_idx
                all_history_features = get_feature(X, unsensed_output, 'adp', 
                                                 self.sensed_locations, self.unsensed_locations, self.scaler)
                
                # Stage 2: Forecast - predict future values for sensed locations
                sensed_future_output = self.forecast_model(all_history_features)  # output shape [bs, T', M, 1]
                
                # Generate output's other features: time-of-day, day-of-week, rank, time_idx
                sensed_future_features = get_feature(X, sensed_future_output, 'forecast', 
                                                   self.sensed_locations, self.unsensed_locations, self.scaler)
                
                # Stage 3: Aggregation - combine history and future for final prediction
                agg_output = self.agg_model(all_history_features, sensed_future_features)  # output shape [bs, T', M', 1]
                
                # Collect results for each stage
                history_unsensed_true.append(unsensed_X[..., :1])
                history_unsensed_pred.append(unsensed_output)
                future_sensed_true.append(sensed_Y[..., :1])
                future_sensed_pred.append(sensed_future_output)
                target_true.append(label)
                target_pred.append(agg_output)
            
            # Concatenate all batch results
            history_unsensed_true = torch.cat(history_unsensed_true, dim=0)
            history_unsensed_pred = torch.cat(history_unsensed_pred, dim=0)
            future_sensed_true = torch.cat(future_sensed_true, dim=0)
            future_sensed_pred = torch.cat(future_sensed_pred, dim=0)
            target_true = torch.cat(target_true, dim=0)
            target_pred = torch.cat(target_pred, dim=0)
            
            # Save predictions for further analysis
            self._save_predictions(target_true, target_pred)
            
            # Log detailed results for each stage
            self._log_stage_results('Adaptation', history_unsensed_pred, history_unsensed_true)
            self._log_stage_results('Forecast', future_sensed_pred, future_sensed_true)
            self._log_stage_results('Aggregation', target_pred, target_true)
            
            self.logger.info('Inference completed successfully!')


