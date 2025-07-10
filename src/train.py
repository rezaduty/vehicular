"""
Training script for autonomous driving perception models
Supports object detection, domain adaptation, and unsupervised learning
"""

import argparse
import yaml
import torch
import torch.nn as nn
import wandb
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Any

from data.dataset_loader import DatasetLoader
from data.transforms import get_transforms
from models.object_detection import YOLOv8Detector, ParallelPatchDetector
from models.domain_adaptation import DomainAdversarialNetwork, DomainAdapter
from unsupervised.lost import SelfSupervisedObjectDetection
from utils.metrics import compute_metrics, log_metrics
from utils.visualization import save_training_plots


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train autonomous driving perception models')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--task', type=str, required=True,
                        choices=['object_detection', 'domain_adaptation', 'unsupervised'],
                        help='Training task')
    parser.add_argument('--dataset', type=str, default='kitti',
                        choices=['kitti', 'carla', 'mixed'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for models and logs')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config with command line arguments"""
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    return config


def setup_device(gpu_id: int = 0) -> torch.device:
    """Setup training device"""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def create_model(task: str, config: Dict[str, Any], device: torch.device):
    """Create model based on task"""
    if task == 'object_detection':
        model = YOLOv8Detector(
            num_classes=config['models']['object_detection']['num_classes'],
            confidence_threshold=config['models']['object_detection']['confidence_threshold'],
            nms_threshold=config['models']['object_detection']['nms_threshold']
        )
        
        # Wrap with parallel patch detector if enabled
        if config['inference']['patch_detection']['enabled']:
            model = ParallelPatchDetector(
                base_detector=model,
                patch_size=tuple(config['inference']['patch_detection']['patch_size']),
                overlap=config['inference']['patch_detection']['overlap'],
                min_object_size=config['inference']['patch_detection']['min_object_size']
            )
    
    elif task == 'domain_adaptation':
        model = DomainAdversarialNetwork(
            num_classes=config['models']['object_detection']['num_classes'],
            lambda_grl=config['models']['domain_adaptation']['lambda_grl']
        )
    
    elif task == 'unsupervised':
        return SelfSupervisedObjectDetection(config)
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return model.to(device)


def create_optimizer(model, config: Dict[str, Any]):
    """Create optimizer"""
    optimizer_name = config['training'].get('optimizer', 'adamw').lower()
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, config: Dict[str, Any]):
    """Create learning rate scheduler"""
    scheduler_name = config['training'].get('scheduler', 'cosine').lower()
    
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['learning_rate'] * 0.01
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['epochs'] // 3,
            gamma=0.1
        )
    elif scheduler_name == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    else:
        scheduler = None
    
    return scheduler


def train_object_detection(model, dataloader, optimizer, scheduler, config, device, logger):
    """Train object detection model"""
    model.train()
    
    for epoch in range(config['training']['epochs']):
        epoch_loss = 0.0
        epoch_metrics = {'precision': 0.0, 'recall': 0.0, 'map': 0.0}
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['images'].to(device)
            targets = batch.get('labels', [])
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(model, 'train_step'):
                loss_dict = model.train_step(images, targets)
                loss = loss_dict['loss']
            else:
                outputs = model(images)
                # Simplified loss calculation for demo
                loss = torch.tensor(0.5, requires_grad=True, device=device)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['gradient_clip']
                )
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{epoch_loss/(batch_idx+1):.4f}'
            })
        
        # Scheduler step
        if scheduler:
            scheduler.step()
        
        # Log metrics
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Log to wandb if enabled
        if wandb.run:
            wandb.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })


def train_domain_adaptation(source_dataloader, target_dataloader, model, config, device, logger):
    """Train domain adaptation model"""
    domain_adapter = DomainAdapter(
        model=model,
        source_dataloader=source_dataloader,
        target_dataloader=target_dataloader,
        config=config
    )
    
    for epoch in range(config['training']['epochs']):
        avg_loss = domain_adapter.train_epoch(epoch)
        
        logger.info(f"Epoch {epoch+1}: Domain Adaptation Loss = {avg_loss:.4f}")
        
        if wandb.run:
            wandb.log({
                'epoch': epoch + 1,
                'domain_loss': avg_loss
            })


def train_unsupervised(model, dataloader, config, logger):
    """Train unsupervised model"""
    for epoch in range(config['training']['epochs']):
        avg_loss = model.train_epoch(dataloader)
        
        logger.info(f"Epoch {epoch+1}: Unsupervised Loss = {avg_loss:.4f}")
        
        if wandb.run:
            wandb.log({
                'epoch': epoch + 1,
                'unsupervised_loss': avg_loss
            })


def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    checkpoint_path = Path(output_dir) / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting training...")
    
    # Load and override config
    config = load_config(args.config)
    config = override_config(config, args)
    
    # Setup device
    device = setup_device(args.gpu)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=config['logging']['wandb']['project'],
            config=config,
            name=f"{args.task}_{args.dataset}"
        )
    
    # Create data loaders
    dataset_loader = DatasetLoader(config)
    
    if args.task == 'domain_adaptation':
        # Load both source and target datasets
        source_dataloader = dataset_loader.load_dataset(
            'carla', 'train', 
            config['training']['batch_size'],
            num_workers=4
        )
        target_dataloader = dataset_loader.load_dataset(
            'kitti', 'train',
            config['training']['batch_size'], 
            num_workers=4
        )
    else:
        # Load single dataset
        if args.dataset == 'mixed':
            dataloader = dataset_loader.load_mixed_dataset(
                ['kitti', 'carla'], 'train',
                config['training']['batch_size'],
                num_workers=4
            )
        else:
            dataloader = dataset_loader.load_dataset(
                args.dataset, 'train',
                config['training']['batch_size'],
                num_workers=4
            )
    
    # Create model
    model = create_model(args.task, config, device)
    logger.info(f"Created {args.task} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and scheduler
    if args.task != 'unsupervised':
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
    else:
        optimizer = scheduler = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.resume)
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # Training loop
    try:
        if args.task == 'object_detection':
            train_object_detection(
                model, dataloader, optimizer, scheduler, config, device, logger
            )
        
        elif args.task == 'domain_adaptation':
            train_domain_adaptation(
                source_dataloader, target_dataloader, model, config, device, logger
            )
        
        elif args.task == 'unsupervised':
            train_unsupervised(model, dataloader, config, logger)
        
        # Save final model
        final_model_path = output_dir / f'{args.task}_final_model.pth'
        if hasattr(model, 'save_model'):
            model.save_model(str(final_model_path))
        else:
            torch.save(model.state_dict(), final_model_path)
        
        logger.info(f"Training completed. Model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save checkpoint
        if optimizer:
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, start_epoch, 0.0, output_dir
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main() 