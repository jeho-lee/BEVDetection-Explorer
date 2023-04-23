# Copyright (c) Phigent Robotics. All rights reserved.

"""

2023-4-8 (ai-datacenter)
- EfficientNetV2-m backbone
- batch_size_per_device: 2
- lr: 2e-4
- find_unused_parameters = True

2023-4-15
- lr = 4e-4
- weight decay = 1e-2

2023-4-20
- lr = 5e-4
- weight decay = 1e-2
- lr_decay_steps = [19, 23]
- warmup iters = 2000

2023-4-23 (final trial)
- lr = 1e-3
- weight decay = 1e-2
- lr_decay_steps = [16, 22]
- warmup_iters = 2000
"""

# GPU and batch size
num_gpu = 8
batch_size_per_device = 2 # effnetv2: max 2

# learning rate and scheduling
lr = 1e-3
lr_decay_steps = [16, 22]

# AdamW
weight_decay = 1e-2 # 1e-2 in bevdet and BEVFormerV2
warmup_iters = 2000 # 500 in bevdet
warmup_ratio = 0.001

_base_ = ['../../_base_/datasets/nus-3d.py', '../../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    # 'depth': [1.0, 60.0, 1.0], # bevdet
    'depth': [2.0, 58.0, 0.5], # BEVDepth
}

# Image backbone checkpoint

# EfficientNet
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b2_3rdparty-ra-noisystudent_in1k_20221103-301ed299.pth'
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b6_3rdparty-ra-noisystudent_in1k_20221103-7de7d2cc.pth'

# EfficientNetV2
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_3rdparty_in21k_20221220-c0572b56.pth' # efficientnetv2-s
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_3rdparty_in21k_20221220-073e944c.pth' # efficientnetv2-m
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-l_3rdparty_in21k_20221220-f28f91e1.pth' # efficientnetv2-l
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-xl_3rdparty_in21k_20221220-b2c9329c.pth' # efficientnetv2-xl

# LeViT
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/levit/levit-192_3rdparty_in1k_20230117-8217a0f9.pth' # 192
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/levit/levit-256_3rdparty_in1k_20230117-5ae2ce7d.pth' # 256
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/levit/levit-384_3rdparty_in1k_20230117-f3539cce.pth' # 384

# Intermediate Checkpointing to save GPU memory.
with_cp = False

voxel_size = [0.1, 0.1, 0.2] # For CenterHead

numC_Trans = 80 # BEV channels

# EfficientNetV2
find_unused_parameters = True

model = dict(
    type='BEVDepth',
    
    # EfficientNetV2 backbone
    img_backbone=dict(
        type='EfficientNetV2',
        arch='m',
        out_indices=[4, 5, 6, 7],
        frozen_stages=0,
        with_cp=with_cp,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[160, 176, 304, 512],
        out_channels=[128, 128, 128, 128],
        upsample_strides=[1, 1, 2, 2]),

    # BEV feature extraction
    img_view_transformer=dict( # to LSSViewTransformerBEVDepth
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        
        # Whether to use deformable convolution (TODO check)
        # In BEVDepth, the default setting is to use dcn
        depthnet_cfg=dict(use_dcn=False, use_aspp=True),
        downsample=16),
    
    # BEV encoding before detection head
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2,
                      numC_Trans * 4,
                      numC_Trans * 8],
        backbone_output_ids=[-1, 0, 1, 2]),
    img_bev_encoder_neck=dict(
        type='SECONDFPN',
        in_channels=[numC_Trans, 160, 320, 640],
        upsample_strides=[1, 2, 4, 8],
        out_channels=[64, 64, 64, 64]),
    
    # BEVDet's detection head
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])), # 0.2 is the velocity code weight (last two elements)
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,

            # Scale-NMS
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])))

# Data
dataset_type = 'NuScenesDataset'
data_root = '/datasets/nuscenes/' # AI datacenter
ann_root = '/home/dlwpgh1994/3D-perception/data/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    
    # Data augmentation
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    
    # Modification for training BEVDet with BEVDepth modules: LiDAR-supervised DepthNet
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth']) # depth supervision
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    data_root=data_root,
    ann_file=ann_root + 'nuscenes_infos_val.pkl')

# Training Config (2023-4-6 by Jeho Lee)
data = dict(
    samples_per_gpu=batch_size_per_device, # If use 8 GPUs, total batch size is 8*8=64
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR')),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train']['dataset'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=lr, weight_decay=weight_decay)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=warmup_ratio,
    step=lr_decay_steps)
runner = dict(type='EpochBasedRunner', max_epochs=24) # 24 epochs

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]