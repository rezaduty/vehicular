"""
TensorFlow/Keras models for autonomous driving perception
Includes RetinaNet, EfficientDet, and DeepLabV3+ implementations
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2


class RetinaNet(Model):
    """RetinaNet implementation using TensorFlow/Keras"""
    
    def __init__(self, 
                 num_classes: int = 10,
                 backbone: str = 'resnet50',
                 input_shape: Tuple[int, int, int] = (384, 1280, 3),
                 anchor_scales: List[float] = [1.0, 1.26, 1.59],
                 anchor_ratios: List[float] = [0.5, 1.0, 2.0],
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.input_shape = input_shape
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        
        # Build backbone
        self.backbone = self._build_backbone()
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork()
        
        # Classification and regression heads
        self.classification_head = ClassificationHead(num_classes, len(anchor_scales) * len(anchor_ratios))
        self.regression_head = RegressionHead(len(anchor_scales) * len(anchor_ratios))
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator(anchor_scales, anchor_ratios)
        
        # Loss functions
        self.focal_loss = FocalLoss()
        self.smooth_l1_loss = SmoothL1Loss()
        
    def _build_backbone(self):
        """Build backbone network"""
        if self.backbone_name == 'resnet50':
            base_model = keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Extract feature maps at different scales
            c3_output = base_model.get_layer('conv3_block4_out').output
            c4_output = base_model.get_layer('conv4_block6_out').output
            c5_output = base_model.get_layer('conv5_block3_out').output
            
            backbone = Model(
                inputs=base_model.input,
                outputs=[c3_output, c4_output, c5_output],
                name='resnet50_backbone'
            )
            
        elif self.backbone_name == 'efficientnet-b0':
            base_model = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Extract appropriate feature maps
            c3_output = base_model.get_layer('block3a_expand_activation').output
            c4_output = base_model.get_layer('block5a_expand_activation').output
            c5_output = base_model.get_layer('block7a_expand_activation').output
            
            backbone = Model(
                inputs=base_model.input,
                outputs=[c3_output, c4_output, c5_output],
                name='efficientnet_backbone'
            )
            
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        return backbone
    
    def call(self, inputs, training=None):
        """Forward pass"""
        # Extract features
        c3, c4, c5 = self.backbone(inputs, training=training)
        
        # Feature pyramid
        fpn_features = self.fpn([c3, c4, c5], training=training)
        
        # Generate anchors
        anchors = self.anchor_generator(fpn_features, tf.shape(inputs)[1:3])
        
        # Predictions
        classifications = []
        regressions = []
        
        for feature in fpn_features:
            cls_output = self.classification_head(feature, training=training)
            reg_output = self.regression_head(feature, training=training)
            
            classifications.append(cls_output)
            regressions.append(reg_output)
        
        return {
            'classifications': classifications,
            'regressions': regressions,
            'anchors': anchors
        }
    
    def compute_loss(self, predictions, targets):
        """Compute RetinaNet loss"""
        classifications = predictions['classifications']
        regressions = predictions['regressions']
        anchors = predictions['anchors']
        
        # Prepare targets
        cls_targets, reg_targets = self._prepare_targets(targets, anchors)
        
        # Classification loss (Focal Loss)
        cls_loss = 0
        for cls_pred, cls_target in zip(classifications, cls_targets):
            cls_loss += self.focal_loss(cls_target, cls_pred)
        
        # Regression loss (Smooth L1)
        reg_loss = 0
        for reg_pred, reg_target in zip(regressions, reg_targets):
            reg_loss += self.smooth_l1_loss(reg_target, reg_pred)
        
        total_loss = cls_loss + reg_loss
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'regression_loss': reg_loss
        }
    
    def _prepare_targets(self, targets, anchors):
        """Prepare targets for loss computation"""
        # This is a simplified version - full implementation would require
        # proper anchor matching and target assignment
        batch_size = tf.shape(targets['boxes'])[0]
        
        # Placeholder implementation
        cls_targets = []
        reg_targets = []
        
        for anchor_level in anchors:
            anchor_shape = tf.shape(anchor_level)
            cls_target = tf.zeros((batch_size, anchor_shape[1], self.num_classes))
            reg_target = tf.zeros((batch_size, anchor_shape[1], 4))
            
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
        
        return cls_targets, reg_targets


class EfficientDet(Model):
    """EfficientDet implementation using TensorFlow/Keras"""
    
    def __init__(self, 
                 num_classes: int = 10,
                 compound_coef: int = 0,
                 input_shape: Tuple[int, int, int] = (384, 1280, 3),
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.compound_coef = compound_coef
        self.input_shape = input_shape
        
        # EfficientNet backbone
        self.backbone = self._build_efficientnet_backbone()
        
        # BiFPN
        self.bifpn = BiFPN(compound_coef)
        
        # Detection heads
        self.classification_head = EfficientDetClassificationHead(num_classes)
        self.regression_head = EfficientDetRegressionHead()
        
    def _build_efficientnet_backbone(self):
        """Build EfficientNet backbone"""
        backbone_name = f'efficientnet-b{self.compound_coef}'
        
        if backbone_name == 'efficientnet-b0':
            base_model = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif backbone_name == 'efficientnet-b1':
            base_model = keras.applications.EfficientNetB1(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            # Fallback to B0
            base_model = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Extract multi-scale features
        feature_layers = [
            'block3a_expand_activation',
            'block5a_expand_activation', 
            'block7a_expand_activation'
        ]
        
        outputs = [base_model.get_layer(layer_name).output for layer_name in feature_layers]
        
        return Model(inputs=base_model.input, outputs=outputs, name='efficientnet_backbone')
    
    def call(self, inputs, training=None):
        """Forward pass"""
        # Extract backbone features
        backbone_features = self.backbone(inputs, training=training)
        
        # BiFPN feature fusion
        fpn_features = self.bifpn(backbone_features, training=training)
        
        # Detection predictions
        classifications = []
        regressions = []
        
        for feature in fpn_features:
            cls_output = self.classification_head(feature, training=training)
            reg_output = self.regression_head(feature, training=training)
            
            classifications.append(cls_output)
            regressions.append(reg_output)
        
        return {
            'classifications': classifications,
            'regressions': regressions
        }


class DeepLabV3PlusTF(Model):
    """DeepLabV3+ implementation using TensorFlow/Keras"""
    
    def __init__(self,
                 num_classes: int = 19,
                 backbone: str = 'resnet50',
                 input_shape: Tuple[int, int, int] = (384, 1280, 3),
                 output_stride: int = 16,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.input_shape = input_shape
        self.output_stride = output_stride
        
        # Build backbone
        self.backbone = self._build_backbone()
        
        # ASPP module
        self.aspp = ASPP(256, output_stride)
        
        # Decoder
        self.decoder = DeepLabV3PlusDecoder(num_classes)
        
    def _build_backbone(self):
        """Build backbone network"""
        if self.backbone_name == 'resnet50':
            base_model = keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Extract low-level and high-level features
            low_level_features = base_model.get_layer('conv2_block3_out').output
            high_level_features = base_model.get_layer('conv5_block3_out').output
            
            backbone = Model(
                inputs=base_model.input,
                outputs=[low_level_features, high_level_features],
                name='resnet50_backbone'
            )
            
        elif self.backbone_name == 'mobilenet_v2':
            base_model = keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            low_level_features = base_model.get_layer('block_3_expand').output
            high_level_features = base_model.get_layer('out_relu').output
            
            backbone = Model(
                inputs=base_model.input,
                outputs=[low_level_features, high_level_features],
                name='mobilenet_v2_backbone'
            )
            
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        return backbone
    
    def call(self, inputs, training=None):
        """Forward pass"""
        # Extract features
        low_level_features, high_level_features = self.backbone(inputs, training=training)
        
        # ASPP
        aspp_features = self.aspp(high_level_features, training=training)
        
        # Decoder
        output = self.decoder([aspp_features, low_level_features], training=training)
        
        # Upsample to input resolution
        output = tf.image.resize(output, tf.shape(inputs)[1:3], method='bilinear')
        
        return output


# Supporting modules
class FeaturePyramidNetwork(layers.Layer):
    """Feature Pyramid Network"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Lateral connections
        self.lateral_conv_c5 = layers.Conv2D(256, 1, padding='same')
        self.lateral_conv_c4 = layers.Conv2D(256, 1, padding='same')
        self.lateral_conv_c3 = layers.Conv2D(256, 1, padding='same')
        
        # Output convolutions
        self.output_conv_p5 = layers.Conv2D(256, 3, padding='same')
        self.output_conv_p4 = layers.Conv2D(256, 3, padding='same')
        self.output_conv_p3 = layers.Conv2D(256, 3, padding='same')
        
        # Additional levels
        self.p6_conv = layers.Conv2D(256, 3, strides=2, padding='same')
        self.p7_conv = layers.Conv2D(256, 3, strides=2, padding='same')
        self.p7_relu = layers.ReLU()
        
    def call(self, inputs, training=None):
        c3, c4, c5 = inputs
        
        # Lateral connections
        p5 = self.lateral_conv_c5(c5)
        p4 = self.lateral_conv_c4(c4) + tf.image.resize(p5, tf.shape(c4)[1:3])
        p3 = self.lateral_conv_c3(c3) + tf.image.resize(p4, tf.shape(c3)[1:3])
        
        # Output convolutions
        p5 = self.output_conv_p5(p5)
        p4 = self.output_conv_p4(p4)
        p3 = self.output_conv_p3(p3)
        
        # Additional levels
        p6 = self.p6_conv(c5)
        p7 = self.p7_conv(self.p7_relu(p6))
        
        return [p3, p4, p5, p6, p7]


class ClassificationHead(layers.Layer):
    """Classification head for RetinaNet"""
    
    def __init__(self, num_classes, num_anchors, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Convolution layers
        self.conv1 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(256, 3, padding='same', activation='relu')
        
        # Output layer
        self.classifier = layers.Conv2D(num_anchors * num_classes, 3, padding='same')
        
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Classification output
        output = self.classifier(x)
        
        # Reshape for loss computation
        batch_size = tf.shape(output)[0]
        output = tf.reshape(output, [batch_size, -1, self.num_classes])
        
        return tf.nn.sigmoid(output)


class RegressionHead(layers.Layer):
    """Regression head for RetinaNet"""
    
    def __init__(self, num_anchors, **kwargs):
        super().__init__(**kwargs)
        self.num_anchors = num_anchors
        
        # Convolution layers
        self.conv1 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(256, 3, padding='same', activation='relu')
        
        # Output layer
        self.regressor = layers.Conv2D(num_anchors * 4, 3, padding='same')
        
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Regression output
        output = self.regressor(x)
        
        # Reshape for loss computation
        batch_size = tf.shape(output)[0]
        output = tf.reshape(output, [batch_size, -1, 4])
        
        return output


class BiFPN(layers.Layer):
    """Bidirectional Feature Pyramid Network"""
    
    def __init__(self, compound_coef, **kwargs):
        super().__init__(**kwargs)
        self.compound_coef = compound_coef
        
        # Number of BiFPN layers
        self.num_layers = 3 + compound_coef
        
        # BiFPN blocks
        self.bifpn_blocks = []
        for i in range(self.num_layers):
            self.bifpn_blocks.append(BiFPNBlock())
    
    def call(self, inputs, training=None):
        features = inputs
        
        for bifpn_block in self.bifpn_blocks:
            features = bifpn_block(features, training=training)
        
        return features


class BiFPNBlock(layers.Layer):
    """Single BiFPN block"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Convolution layers for feature fusion
        self.conv_up = layers.Conv2D(256, 3, padding='same')
        self.conv_down = layers.Conv2D(256, 3, padding='same')
        
        # Batch normalization
        self.bn_up = layers.BatchNormalization()
        self.bn_down = layers.BatchNormalization()
        
        # Activation
        self.activation = layers.ReLU()
        
    def call(self, inputs, training=None):
        # Simplified BiFPN implementation
        # In practice, this would involve more complex feature fusion
        
        # Top-down pathway
        features_up = []
        for i, feature in enumerate(inputs):
            if i == 0:
                features_up.append(feature)
            else:
                upsampled = tf.image.resize(features_up[-1], tf.shape(feature)[1:3])
                fused = self.conv_up(feature + upsampled)
                fused = self.bn_up(fused, training=training)
                fused = self.activation(fused)
                features_up.append(fused)
        
        # Bottom-up pathway
        features_down = []
        for i, feature in enumerate(reversed(features_up)):
            if i == 0:
                features_down.append(feature)
            else:
                downsampled = layers.MaxPooling2D(2)(features_down[-1])
                fused = self.conv_down(feature + downsampled)
                fused = self.bn_down(fused, training=training)
                fused = self.activation(fused)
                features_down.append(fused)
        
        return list(reversed(features_down))


class EfficientDetClassificationHead(layers.Layer):
    """Classification head for EfficientDet"""
    
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # Separable convolutions
        self.separable_conv1 = layers.SeparableConv2D(256, 3, padding='same')
        self.separable_conv2 = layers.SeparableConv2D(256, 3, padding='same')
        self.separable_conv3 = layers.SeparableConv2D(256, 3, padding='same')
        
        # Batch normalization
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        
        # Activation
        self.activation = layers.ReLU()
        
        # Output layer
        self.classifier = layers.Conv2D(num_classes * 9, 3, padding='same')  # 9 anchors
        
    def call(self, inputs, training=None):
        x = self.separable_conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        
        x = self.separable_conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        
        x = self.separable_conv3(x)
        x = self.bn3(x, training=training)
        x = self.activation(x)
        
        # Classification output
        output = self.classifier(x)
        
        # Reshape and apply sigmoid
        batch_size = tf.shape(output)[0]
        output = tf.reshape(output, [batch_size, -1, self.num_classes])
        
        return tf.nn.sigmoid(output)


class EfficientDetRegressionHead(layers.Layer):
    """Regression head for EfficientDet"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Separable convolutions
        self.separable_conv1 = layers.SeparableConv2D(256, 3, padding='same')
        self.separable_conv2 = layers.SeparableConv2D(256, 3, padding='same')
        self.separable_conv3 = layers.SeparableConv2D(256, 3, padding='same')
        
        # Batch normalization
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        
        # Activation
        self.activation = layers.ReLU()
        
        # Output layer
        self.regressor = layers.Conv2D(4 * 9, 3, padding='same')  # 4 coordinates * 9 anchors
        
    def call(self, inputs, training=None):
        x = self.separable_conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        
        x = self.separable_conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        
        x = self.separable_conv3(x)
        x = self.bn3(x, training=training)
        x = self.activation(x)
        
        # Regression output
        output = self.regressor(x)
        
        # Reshape
        batch_size = tf.shape(output)[0]
        output = tf.reshape(output, [batch_size, -1, 4])
        
        return output


class ASPP(layers.Layer):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, filters, output_stride, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
        # Atrous rates based on output stride
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            atrous_rates = [6, 12, 18]
        
        # ASPP branches
        self.conv_1x1 = layers.Conv2D(filters, 1, padding='same')
        self.conv_3x3_1 = layers.Conv2D(filters, 3, padding='same', dilation_rate=atrous_rates[0])
        self.conv_3x3_2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=atrous_rates[1])
        self.conv_3x3_3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=atrous_rates[2])
        
        # Global average pooling branch
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.conv_1x1_pool = layers.Conv2D(filters, 1, padding='same')
        
        # Batch normalization
        self.bn_1x1 = layers.BatchNormalization()
        self.bn_3x3_1 = layers.BatchNormalization()
        self.bn_3x3_2 = layers.BatchNormalization()
        self.bn_3x3_3 = layers.BatchNormalization()
        self.bn_pool = layers.BatchNormalization()
        
        # Activation
        self.activation = layers.ReLU()
        
        # Output projection
        self.output_conv = layers.Conv2D(filters, 1, padding='same')
        self.output_bn = layers.BatchNormalization()
        
    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        
        # ASPP branches
        branch_1x1 = self.conv_1x1(inputs)
        branch_1x1 = self.bn_1x1(branch_1x1, training=training)
        branch_1x1 = self.activation(branch_1x1)
        
        branch_3x3_1 = self.conv_3x3_1(inputs)
        branch_3x3_1 = self.bn_3x3_1(branch_3x3_1, training=training)
        branch_3x3_1 = self.activation(branch_3x3_1)
        
        branch_3x3_2 = self.conv_3x3_2(inputs)
        branch_3x3_2 = self.bn_3x3_2(branch_3x3_2, training=training)
        branch_3x3_2 = self.activation(branch_3x3_2)
        
        branch_3x3_3 = self.conv_3x3_3(inputs)
        branch_3x3_3 = self.bn_3x3_3(branch_3x3_3, training=training)
        branch_3x3_3 = self.activation(branch_3x3_3)
        
        # Global average pooling branch
        branch_pool = self.global_avg_pool(inputs)
        branch_pool = self.conv_1x1_pool(branch_pool)
        branch_pool = self.bn_pool(branch_pool, training=training)
        branch_pool = self.activation(branch_pool)
        branch_pool = tf.image.resize(branch_pool, input_shape[1:3])
        
        # Concatenate all branches
        output = tf.concat([branch_1x1, branch_3x3_1, branch_3x3_2, branch_3x3_3, branch_pool], axis=-1)
        
        # Output projection
        output = self.output_conv(output)
        output = self.output_bn(output, training=training)
        output = self.activation(output)
        
        return output


class DeepLabV3PlusDecoder(layers.Layer):
    """DeepLabV3+ Decoder"""
    
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # Low-level feature processing
        self.low_level_conv = layers.Conv2D(48, 1, padding='same')
        self.low_level_bn = layers.BatchNormalization()
        
        # Decoder convolutions
        self.decoder_conv1 = layers.Conv2D(256, 3, padding='same')
        self.decoder_bn1 = layers.BatchNormalization()
        self.decoder_conv2 = layers.Conv2D(256, 3, padding='same')
        self.decoder_bn2 = layers.BatchNormalization()
        
        # Output layer
        self.output_conv = layers.Conv2D(num_classes, 1, padding='same')
        
        # Activation
        self.activation = layers.ReLU()
        
    def call(self, inputs, training=None):
        aspp_features, low_level_features = inputs
        
        # Process low-level features
        low_level = self.low_level_conv(low_level_features)
        low_level = self.low_level_bn(low_level, training=training)
        low_level = self.activation(low_level)
        
        # Upsample ASPP features
        aspp_upsampled = tf.image.resize(aspp_features, tf.shape(low_level_features)[1:3])
        
        # Concatenate
        concat_features = tf.concat([aspp_upsampled, low_level], axis=-1)
        
        # Decoder convolutions
        x = self.decoder_conv1(concat_features)
        x = self.decoder_bn1(x, training=training)
        x = self.activation(x)
        
        x = self.decoder_conv2(x)
        x = self.decoder_bn2(x, training=training)
        x = self.activation(x)
        
        # Output
        output = self.output_conv(x)
        
        return output


class AnchorGenerator:
    """Anchor generator for object detection"""
    
    def __init__(self, scales, ratios):
        self.scales = scales
        self.ratios = ratios
        
    def __call__(self, feature_maps, image_shape):
        """Generate anchors for all feature maps"""
        anchors = []
        
        for feature_map in feature_maps:
            feature_shape = tf.shape(feature_map)
            anchors_level = self._generate_anchors_for_level(
                feature_shape[1:3], image_shape
            )
            anchors.append(anchors_level)
        
        return anchors
    
    def _generate_anchors_for_level(self, feature_shape, image_shape):
        """Generate anchors for a single feature level"""
        height, width = feature_shape[0], feature_shape[1]
        
        # Create coordinate grids
        y_coords = tf.range(height, dtype=tf.float32)
        x_coords = tf.range(width, dtype=tf.float32)
        
        x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
        
        # Flatten grids
        x_grid = tf.reshape(x_grid, [-1])
        y_grid = tf.reshape(y_grid, [-1])
        
        # Generate anchors for each scale and ratio
        anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                # Calculate anchor dimensions
                anchor_width = scale * tf.sqrt(ratio)
                anchor_height = scale / tf.sqrt(ratio)
                
                # Create anchors
                x1 = x_grid - anchor_width / 2
                y1 = y_grid - anchor_height / 2
                x2 = x_grid + anchor_width / 2
                y2 = y_grid + anchor_height / 2
                
                anchor = tf.stack([x1, y1, x2, y2], axis=1)
                anchors.append(anchor)
        
        return tf.concat(anchors, axis=0)


class FocalLoss(keras.losses.Loss):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        
    def call(self, y_true, y_pred):
        # Compute focal loss
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Compute cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Compute focal weight
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = self.alpha * tf.pow(1 - pt, self.gamma)
        
        # Compute focal loss
        focal_loss = focal_weight * ce
        
        return tf.reduce_sum(focal_loss, axis=-1)


class SmoothL1Loss(keras.losses.Loss):
    """Smooth L1 Loss for bounding box regression"""
    
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        
    def call(self, y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        
        # Smooth L1 loss
        loss = tf.where(
            tf.less(diff, self.beta),
            0.5 * tf.square(diff) / self.beta,
            diff - 0.5 * self.beta
        )
        
        return tf.reduce_sum(loss, axis=-1)


# Model factory functions
def create_retinanet(num_classes=10, backbone='resnet50', input_shape=(384, 1280, 3)):
    """Create RetinaNet model"""
    return RetinaNet(
        num_classes=num_classes,
        backbone=backbone,
        input_shape=input_shape
    )


def create_efficientdet(num_classes=10, compound_coef=0, input_shape=(384, 1280, 3)):
    """Create EfficientDet model"""
    return EfficientDet(
        num_classes=num_classes,
        compound_coef=compound_coef,
        input_shape=input_shape
    )


def create_deeplabv3plus(num_classes=19, backbone='resnet50', input_shape=(384, 1280, 3)):
    """Create DeepLabV3+ model"""
    return DeepLabV3PlusTF(
        num_classes=num_classes,
        backbone=backbone,
        input_shape=input_shape
    ) 