import torch
import os
import copy
from tqdm import tqdm
from lib.logger import get_logger
from lib.metrics import All_Metrics
from lib.dataloader import get_feature

class Trainer(object):
    def __init__(self, sensed_locations, unsensed_locations, adp_model, forecast_model, agg_model,
                 adp_optimizer, forecast_optimizer, agg_optimizer, train_loader, val_loader, test_loader,
                 scaler, args, device, loss):
        super(Trainer, self).__init__()
        self.sensed_locations = sensed_locations
        self.unsensed_locations = unsensed_locations
        self.adp_model = adp_model
        self.forecast_model = forecast_model
        self.agg_model = agg_model
        self.adp_optimizer = adp_optimizer
        self.forecast_optimizer = forecast_optimizer
        self.agg_optimizer = agg_optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.device = device
        self.loss = loss
        self.epochs = args.epochs
        self.early_stop_patience = args.early_stop_patience

        # Paths for saving best models
        self.adp_best_path = os.path.join(args.log_dir, 'adp_best_model.pth')
        self.forecast_best_path = os.path.join(args.log_dir, 'forecast_best_model.pth')
        self.agg_best_path = os.path.join(args.log_dir, 'agg_best_model.pth')

        # Logger setup
        os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, debug=False)
        self.logger.info(f'Experiment log path in: {args.log_dir}')

    def _log_metrics(self, epoch, true, pred, stage):
        """Compute the metrics for the given stage."""
        mae, rmse, mape, _ = All_Metrics(pred, true, self.args.mae_thresh, self.args.mape_thresh)
        if stage == 'train':
            self.logger.info(f'[TRAIN][{self.current_stage}] Epoch {epoch}: MAE={mae:.4f}')
        elif stage == 'val':
            self.logger.info(f'[VAL][{self.current_stage}] Epoch {epoch}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape*100:.2f}%')
        else:  # test
            self.logger.info('**************** TEST RESULT ****************')
            for t in range(true.shape[1]):
                mae_t, rmse_t, mape_t, _ = All_Metrics(pred[:, t, ...], true[:, t, ...], self.args.mae_thresh, self.args.mape_thresh)
                self.logger.info(f'Horizon {t+1:02d}: MAE={mae_t:.2f}, RMSE={rmse_t:.2f}, MAPE={mape_t*100:.2f}%')
            self.logger.info(f'Average Horizon: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape*100:.2f}%')
        return mae

    def _save_best_model(self, val_loss):
        """Save model if validation loss improves, implement early stopping."""
        stop = False
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_count = 0
            self.logger.info(f'[{self.current_stage.upper()}] New best model saved.')
            if self.current_stage == 'adp':
                torch.save(self.adp_model.state_dict(), self.adp_best_path)
            elif self.current_stage == 'forecast':
                torch.save(self.forecast_model.state_dict(), self.forecast_best_path)
            else:
                torch.save(self.agg_model.state_dict(), self.agg_best_path)
        else:
            self.patience_count += 1
            if self.patience_count >= self.early_stop_patience:
                stop = True
        return stop

    def _load_model_safe(self, path):
        """Safely load model state dict."""
        # Convert int device to string
        if isinstance(self.device, int):
            device_str = f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu'
            return torch.load(path, map_location=device_str)
        return torch.load(path, map_location=self.device)

    def train_epoch(self, epoch):
        """Train one epoch for the current stage."""
        total_loss = 0
        stage_dict = {
            'adp': (self.adp_model, self.adp_optimizer, self.adp_model.parameters()),
            'forecast': (self.forecast_model, self.forecast_optimizer, self.forecast_model.parameters()),
            'agg': (self.agg_model, self.agg_optimizer, self.agg_model.parameters())
        }
        self.adp_model.train()
        self.forecast_model.train()
        self.agg_model.train()

        for count, batch in enumerate(self.train_loader):
            X, unsensed_X, sensed_Y, target = batch 
            #history sensed; history unsensed; future sensed; future unsensed
            # X shape [bs, T, M, C] bs=64, T=12, M= # of sensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
            # unsensed_X shape [bs, T, M', C] bs=64, T=12, M'= # of unsensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
            # sensed_Y shape [bs, T', M, C] bs=64, T'=96, M= # of sensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
            # target shape [bs, T', M', C] bs=64, T'=96, M'= # of unsensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
            label = target[..., :self.args.output_dim]
            model, optimizer, params = stage_dict[self.current_stage]
            optimizer.zero_grad()

            if self.current_stage == 'adp':
                output = model(X[..., :-1])
                loss = self.loss(output, unsensed_X[..., :1])
            elif self.current_stage == 'forecast':
                unsensed_output = self.adp_model(X[..., :-1])
                all_history_features = get_feature(X, unsensed_output, 'adp', self.sensed_locations, self.unsensed_locations, self.scaler)
                output = model(all_history_features)
                loss = self.loss(output, sensed_Y[..., :1])
            else:  # agg
                unsensed_output = self.adp_model(X[..., :-1])
                all_history_features = get_feature(X, unsensed_output, 'adp', self.sensed_locations, self.unsensed_locations, self.scaler)
                sensed_future_output = self.forecast_model(all_history_features)
                sensed_future_features = get_feature(X, sensed_future_output, 'forecast', self.sensed_locations, self.unsensed_locations, self.scaler)
                output = model(all_history_features, sensed_future_features)
                loss = self.loss(output, label)

            loss.backward()
            total_loss += loss.item()
            torch.cuda.empty_cache()
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(params, self.args.max_grad_norm)
            optimizer.step()

        avg_loss = total_loss / (count + 1)
        self.logger.info(f'[TRAIN][{self.current_stage}] Epoch {epoch}: Avg Loss={avg_loss:.6f}')

    def val_epoch(self, epoch):
        """Validate one epoch for the current stage."""
        self.adp_model.eval()
        self.forecast_model.eval()
        self.agg_model.eval()

        pred_list, true_list = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                X, unsensed_X, sensed_Y, target = batch
                #history sensed; history unsensed; future sensed; future unsensed
                # X shape [bs, T, M, C] bs=64, T=12, M= # of sensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
                # unsensed_X shape [bs, T, M', C] bs=64, T=12, M'= # of unsensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
                # sensed_Y shape [bs, T', M, C] bs=64, T'=96, M= # of sensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)
                # target shape [bs, T', M', C] bs=64, T'=96, M'= # of unsensed locations, C=5 (data, time-of-day, day-of-week, order, time_idx)


                label = target[..., :self.args.output_dim]

                if self.current_stage == 'adp':
                    unsensed_output = self.adp_model(X[..., :-1])
                    true_list.append(unsensed_X[..., :1])
                    pred_list.append(unsensed_output)
                elif self.current_stage == 'forecast':
                    unsensed_output = self.adp_model(X[..., :-1])
                    all_history_features = get_feature(X, unsensed_output, 'adp', self.sensed_locations, self.unsensed_locations, self.scaler)
                    sensed_future_output = self.forecast_model(all_history_features)
                    true_list.append(sensed_Y[..., :1])
                    pred_list.append(sensed_future_output)
                else:  # agg
                    unsensed_output = self.adp_model(X[..., :-1])
                    all_history_features = get_feature(X, unsensed_output, 'adp', self.sensed_locations, self.unsensed_locations, self.scaler)
                    sensed_future_output = self.forecast_model(all_history_features)
                    sensed_future_features = get_feature(X, sensed_future_output, 'forecast', self.sensed_locations, self.unsensed_locations, self.scaler)
                    agg_output = self.agg_model(all_history_features, sensed_future_features)
                    true_list.append(label)
                    pred_list.append(agg_output)

        true_tensor = torch.cat(true_list, dim=0)
        pred_tensor = torch.cat(pred_list, dim=0)
        val_loss = self._log_metrics(epoch, true_tensor, pred_tensor, 'val')
        should_stop = self._save_best_model(val_loss)
        return should_stop

    def test(self):
        """Run the full three-stage pipeline and evaluate on the test set."""
        self.logger.info('Starting final evaluation with complete three-stage pipeline...')
        self.adp_model.eval()
        self.forecast_model.eval()
        self.agg_model.eval()
        target_predictions, target_ground_truth = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Final Testing'):
                X, unsensed_X, sensed_Y, target = batch
                label = target[..., :self.args.output_dim]
                unsensed_output = self.adp_model(X[..., :-1])
                all_history_features = get_feature(X, unsensed_output, 'adp', self.sensed_locations, self.unsensed_locations, self.scaler)
                sensed_future_output = self.forecast_model(all_history_features)
                sensed_future_features = get_feature(X, sensed_future_output, 'forecast', self.sensed_locations, self.unsensed_locations, self.scaler)
                agg_output = self.agg_model(all_history_features, sensed_future_features)
                target_ground_truth.append(label)
                target_predictions.append(agg_output)

        true_tensor = torch.cat(target_ground_truth, dim=0)
        pred_tensor = torch.cat(target_predictions, dim=0)
        self._log_metrics(None, true_tensor, pred_tensor, 'test')

    def train(self):
        """Run the three-stage training pipeline with early stopping and evaluation."""
        self.logger.info('Starting three-stage training pipeline...')
        stages = [
            ('adp', self.adp_model, self.adp_best_path, None),
            ('forecast', self.forecast_model, self.forecast_best_path, self.adp_best_path),
            ('agg', self.agg_model, self.agg_best_path, self.forecast_best_path)
        ]

        for stage_name, model, best_path, load_path in stages:
            self.logger.info(f'Starting {stage_name.upper()} stage training...')
            self.current_stage = stage_name
            self.best_loss = float('inf')
            self.patience_count = 0
            epoch = 0

            # Load previous stage best model if needed
            if load_path is not None and os.path.exists(load_path):
                model.load_state_dict(self._load_model_safe(load_path))
                self.logger.info(f'Loaded parameters from {load_path}')

            while epoch < self.epochs:
                epoch += 1
                self.train_epoch(epoch)
                should_stop = self.val_epoch(epoch)
                if should_stop:
                    self.logger.info(f'Early stopping triggered for {stage_name} at epoch {epoch}')
                    break
            self.logger.info(f'Completed {stage_name.upper()} stage training')

        # Load best aggregation model for final test
        if os.path.exists(self.agg_best_path):
            self.agg_model.load_state_dict(self._load_model_safe(self.agg_best_path))
            self.logger.info('Loaded final aggregation model for testing')
        self.test()
        self.logger.info('Three-stage training pipeline completed successfully!')