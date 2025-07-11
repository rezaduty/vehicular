"""
Comprehensive Domain Adaptation Pipeline for CARLA to KITTI
Integrates multiple domain adaptation techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

from .domain_adaptation import DomainAdversarialNetwork
from .pytorch_models import CORALDomainAdaptation, MMDDomainAdaptation, AdvancedDomainAdversarialNetwork
from .tensorflow_models import create_retinanet, create_efficientdet
from .object_detection import YOLOv8Detector
from ..data.dataset_loader import DatasetLoader


class DomainAdaptationPipeline:
    """Comprehensive domain adaptation pipeline"""
    
    def __init__(self, 
                 config: Dict,
                 source_domain: str = 'carla',
                 target_domain: str = 'kitti',
                 adaptation_methods: List[str] = ['dann', 'coral', 'mmd'],
                 ensemble_method: str = 'weighted_average'):
        
        self.config = config
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.adaptation_methods = adaptation_methods
        self.ensemble_method = ensemble_method
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Initialize optimizers
        self.optimizers = self._initialize_optimizers()
        
        # Initialize schedulers
        self.schedulers = self._initialize_schedulers()
        
        # Loss weights
        self.loss_weights = {
            'task_loss': 1.0,
            'domain_loss': 1.0,
            'coral_loss': 1.0,
            'mmd_loss': 1.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _initialize_models(self) -> Dict[str, nn.Module]:
        """Initialize domain adaptation models"""
        models = {}
        
        num_classes = self.config['models']['object_detection']['num_classes']
        
        # DANN model
        if 'dann' in self.adaptation_methods:
            models['dann'] = AdvancedDomainAdversarialNetwork(
                num_classes=num_classes,
                num_domains=2,  # Source and target
                backbone='resnet50',
                lambda_grl=self.config['models']['domain_adaptation']['lambda_grl']
            )
        
        # CORAL model
        if 'coral' in self.adaptation_methods:
            models['coral'] = CORALDomainAdaptation(
                backbone='resnet50',
                num_classes=num_classes,
                coral_weight=1.0
            )
        
        # MMD model
        if 'mmd' in self.adaptation_methods:
            models['mmd'] = MMDDomainAdaptation(
                backbone='resnet50',
                num_classes=num_classes,
                mmd_weight=1.0
            )
        
        # YOLOv8 baseline
        models['yolov8'] = YOLOv8Detector(
            num_classes=num_classes,
            confidence_threshold=self.config['models']['object_detection']['confidence_threshold']
        )
        
        return models
    
    def _initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Initialize optimizers for each model"""
        optimizers = {}
        
        for name, model in self.models.items():
            if name == 'yolov8':
                # YOLOv8 has its own optimizer handling
                continue
                
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            optimizers[name] = optimizer
        
        return optimizers
    
    def _initialize_schedulers(self) -> Dict[str, torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate schedulers"""
        schedulers = {}
        
        for name, optimizer in self.optimizers.items():
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=self.config['training']['learning_rate'] * 0.01
            )
            schedulers[name] = scheduler
        
        return schedulers
    
    def train_domain_adaptation(self, 
                               source_dataloader: DataLoader,
                               target_dataloader: DataLoader,
                               val_dataloader: DataLoader = None,
                               num_epochs: int = None) -> Dict[str, List[float]]:
        """Train domain adaptation models"""
        
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'domain_accuracy': [],
            'task_accuracy': []
        }
        
        # Set models to training mode
        for model in self.models.values():
            if hasattr(model, 'train'):
                model.train()
        
        self.logger.info(f"Starting domain adaptation training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_losses = self._train_epoch(source_dataloader, target_dataloader, epoch)
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self._validate_epoch(val_dataloader, epoch)
                history['val_loss'].append(val_metrics['loss'])
                history['task_accuracy'].append(val_metrics['accuracy'])
            
            # Update learning rates
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_losses['total']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch + 1)
        
        return history
    
    def _train_epoch(self, source_dataloader: DataLoader, target_dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        total_losses = {method: 0.0 for method in self.adaptation_methods}
        total_losses['total'] = 0.0
        
        # Create iterator for target data
        target_iter = iter(target_dataloader)
        
        for batch_idx, source_batch in enumerate(source_dataloader):
            # Get target batch
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_dataloader)
                target_batch = next(target_iter)
            
            # Move to device
            source_images = source_batch['image'].cuda()
            source_labels = source_batch['labels'].cuda()
            target_images = target_batch['image'].cuda()
            
            # Domain labels
            batch_size = source_images.size(0)
            source_domain_labels = torch.zeros(batch_size, dtype=torch.long).cuda()
            target_domain_labels = torch.ones(batch_size, dtype=torch.long).cuda()
            
            # Train each model
            batch_losses = {}
            
            # DANN training
            if 'dann' in self.adaptation_methods:
                dann_loss = self._train_dann_step(
                    source_images, source_labels, target_images,
                    source_domain_labels, target_domain_labels
                )
                batch_losses['dann'] = dann_loss
            
            # CORAL training
            if 'coral' in self.adaptation_methods:
                coral_loss = self._train_coral_step(
                    source_images, source_labels, target_images
                )
                batch_losses['coral'] = coral_loss
            
            # MMD training
            if 'mmd' in self.adaptation_methods:
                mmd_loss = self._train_mmd_step(
                    source_images, source_labels, target_images
                )
                batch_losses['mmd'] = mmd_loss
            
            # Accumulate losses
            for method, loss in batch_losses.items():
                total_losses[method] += loss
            
            total_losses['total'] += sum(batch_losses.values())
        
        # Average losses
        num_batches = len(source_dataloader)
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return total_losses
    
    def _train_dann_step(self, source_images, source_labels, target_images, 
                        source_domain_labels, target_domain_labels) -> float:
        """Train DANN model for one step"""
        
        model = self.models['dann']
        optimizer = self.optimizers['dann']
        
        optimizer.zero_grad()
        
        # Forward pass on source data
        source_outputs = model(source_images, source_domain_labels)
        
        # Forward pass on target data
        target_outputs = model(target_images, target_domain_labels)
        
        # Task loss (classification on source)
        task_loss = F.cross_entropy(source_outputs['task_predictions'], source_labels)
        
        # Domain loss (discrimination on both domains)
        if isinstance(source_outputs['domain_predictions'], list):
            # Multiple discriminators
            domain_loss = 0
            for i, domain_pred in enumerate(source_outputs['domain_predictions']):
                domain_loss += F.cross_entropy(domain_pred, source_domain_labels)
            
            for i, domain_pred in enumerate(target_outputs['domain_predictions']):
                domain_loss += F.cross_entropy(domain_pred, target_domain_labels)
            
            domain_loss /= len(source_outputs['domain_predictions'])
        else:
            # Single discriminator
            domain_loss = (F.cross_entropy(source_outputs['domain_predictions'], source_domain_labels) +
                          F.cross_entropy(target_outputs['domain_predictions'], target_domain_labels)) / 2
        
        # Total loss
        total_loss = self.loss_weights['task_loss'] * task_loss + self.loss_weights['domain_loss'] * domain_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    
    def _train_coral_step(self, source_images, source_labels, target_images) -> float:
        """Train CORAL model for one step"""
        
        model = self.models['coral']
        optimizer = self.optimizers['coral']
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(source_images, target_images)
        
        # Task loss
        task_loss = F.cross_entropy(outputs['predictions'], source_labels)
        
        # CORAL loss
        coral_loss = outputs['coral_loss']
        
        # Total loss
        total_loss = self.loss_weights['task_loss'] * task_loss + self.loss_weights['coral_loss'] * coral_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    
    def _train_mmd_step(self, source_images, source_labels, target_images) -> float:
        """Train MMD model for one step"""
        
        model = self.models['mmd']
        optimizer = self.optimizers['mmd']
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(source_images, target_images)
        
        # Task loss
        task_loss = F.cross_entropy(outputs['predictions'], source_labels)
        
        # MMD loss
        mmd_loss = outputs['mmd_loss']
        
        # Total loss
        total_loss = self.loss_weights['task_loss'] * task_loss + self.loss_weights['mmd_loss'] * mmd_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    
    def _validate_epoch(self, val_dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate models for one epoch"""
        
        # Set models to evaluation mode
        for model in self.models.values():
            if hasattr(model, 'eval'):
                model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['image'].cuda()
                labels = batch['labels'].cuda()
                
                # Get ensemble predictions
                predictions = self._get_ensemble_predictions(images)
                
                # Compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(predictions.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        # Set models back to training mode
        for model in self.models.values():
            if hasattr(model, 'train'):
                model.train()
        
        return {
            'loss': total_loss / len(val_dataloader),
            'accuracy': correct_predictions / total_predictions
        }
    
    def _get_ensemble_predictions(self, images: torch.Tensor) -> torch.Tensor:
        """Get ensemble predictions from multiple models"""
        
        predictions = []
        
        for method in self.adaptation_methods:
            if method == 'dann':
                outputs = self.models['dann'](images)
                pred = outputs['task_predictions']
            elif method == 'coral':
                outputs = self.models['coral'](images)
                pred = outputs['predictions']
            elif method == 'mmd':
                outputs = self.models['mmd'](images)
                pred = outputs['predictions']
            else:
                continue
            
            predictions.append(F.softmax(pred, dim=1))
        
        if self.ensemble_method == 'weighted_average':
            # Simple average for now
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        elif self.ensemble_method == 'max_voting':
            # Max voting
            ensemble_pred = torch.stack(predictions).max(dim=0)[0]
        else:
            # Default to average
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred
    
    def evaluate_domain_adaptation(self, test_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate domain adaptation performance"""
        
        # Set models to evaluation mode
        for model in self.models.values():
            if hasattr(model, 'eval'):
                model.eval()
        
        metrics = {}
        
        # Evaluate each model individually
        for method in self.adaptation_methods:
            model = self.models[method]
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in test_dataloader:
                    images = batch['image'].cuda()
                    labels = batch['labels'].cuda()
                    
                    if method == 'dann':
                        outputs = model(images)
                        predictions = outputs['task_predictions']
                    elif method == 'coral':
                        outputs = model(images)
                        predictions = outputs['predictions']
                    elif method == 'mmd':
                        outputs = model(images)
                        predictions = outputs['predictions']
                    
                    _, predicted = torch.max(predictions.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            metrics[f'{method}_accuracy'] = correct / total
        
        # Evaluate ensemble
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                images = batch['image'].cuda()
                labels = batch['labels'].cuda()
                
                predictions = self._get_ensemble_predictions(images)
                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        metrics['ensemble_accuracy'] = correct / total
        
        return metrics
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoints"""
        
        checkpoint_dir = Path(self.config.get('output_dir', 'outputs')) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for method, model in self.models.items():
            if method == 'yolov8':
                continue  # YOLOv8 handles its own checkpoints
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizers[method].state_dict(),
                'scheduler_state_dict': self.schedulers[method].state_dict(),
                'config': self.config
            }
            
            checkpoint_path = checkpoint_dir / f'{method}_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str, method: str):
        """Load model checkpoint"""
        
        checkpoint = torch.load(checkpoint_path)
        
        if method in self.models:
            self.models[method].load_state_dict(checkpoint['model_state_dict'])
            
            if method in self.optimizers:
                self.optimizers[method].load_state_dict(checkpoint['optimizer_state_dict'])
            
            if method in self.schedulers:
                self.schedulers[method].load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint for {method} from {checkpoint_path}")
    
    def predict_with_domain_adaptation(self, images: torch.Tensor, use_ensemble: bool = True) -> Dict:
        """Make predictions using domain adaptation models"""
        
        # Set models to evaluation mode
        for model in self.models.values():
            if hasattr(model, 'eval'):
                model.eval()
        
        with torch.no_grad():
            if use_ensemble:
                predictions = self._get_ensemble_predictions(images)
                confidence_scores = torch.max(F.softmax(predictions, dim=1), dim=1)[0]
                predicted_classes = torch.argmax(predictions, dim=1)
            else:
                # Use best performing model (can be determined from validation)
                best_model = self.models[self.adaptation_methods[0]]  # Simplified
                
                if self.adaptation_methods[0] == 'dann':
                    outputs = best_model(images)
                    predictions = outputs['task_predictions']
                elif self.adaptation_methods[0] == 'coral':
                    outputs = best_model(images)
                    predictions = outputs['predictions']
                elif self.adaptation_methods[0] == 'mmd':
                    outputs = best_model(images)
                    predictions = outputs['predictions']
                
                confidence_scores = torch.max(F.softmax(predictions, dim=1), dim=1)[0]
                predicted_classes = torch.argmax(predictions, dim=1)
        
        return {
            'predictions': predicted_classes.cpu().numpy(),
            'confidence_scores': confidence_scores.cpu().numpy(),
            'raw_predictions': predictions.cpu().numpy()
        }


class DomainAdaptationTrainer:
    """Trainer specifically for domain adaptation"""
    
    def __init__(self, pipeline: DomainAdaptationPipeline):
        self.pipeline = pipeline
        self.logger = logging.getLogger(__name__)
    
    def train_carla_to_kitti(self, 
                           carla_dataloader: DataLoader,
                           kitti_dataloader: DataLoader,
                           val_dataloader: DataLoader = None) -> Dict:
        """Train domain adaptation from CARLA to KITTI"""
        
        self.logger.info("Starting CARLA to KITTI domain adaptation training")
        
        # Training history
        history = self.pipeline.train_domain_adaptation(
            source_dataloader=carla_dataloader,
            target_dataloader=kitti_dataloader,
            val_dataloader=val_dataloader
        )
        
        # Final evaluation
        if val_dataloader is not None:
            final_metrics = self.pipeline.evaluate_domain_adaptation(val_dataloader)
            self.logger.info(f"Final evaluation metrics: {final_metrics}")
            history['final_metrics'] = final_metrics
        
        return history
    
    def evaluate_on_real_world_data(self, test_dataloader: DataLoader) -> Dict:
        """Evaluate adapted models on real-world data"""
        
        self.logger.info("Evaluating domain adaptation on real-world data")
        
        metrics = self.pipeline.evaluate_domain_adaptation(test_dataloader)
        
        # Log results
        for method, accuracy in metrics.items():
            self.logger.info(f"{method}: {accuracy:.4f}")
        
        return metrics


# Factory function
def create_domain_adaptation_pipeline(config: Dict, 
                                    adaptation_methods: List[str] = ['dann', 'coral', 'mmd']) -> DomainAdaptationPipeline:
    """Create domain adaptation pipeline"""
    
    pipeline = DomainAdaptationPipeline(
        config=config,
        source_domain='carla',
        target_domain='kitti',
        adaptation_methods=adaptation_methods
    )
    
    return pipeline 