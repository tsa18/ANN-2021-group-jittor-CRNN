common_config = {
    'data_dir': '../data/mnt/ramdisk/max/90kDICT32px/',
    'img_width': 100,
    'img_height': 32,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
}

train_config = {
    'epochs': 100,
    'train_batch_size': 128,
    'eval_batch_size': 512,
    'lr': 5e-5,
    'show_interval': 100,
    'valid_interval': 2000,
    'save_interval': 2000,
    'cpu_workers': 16,
    'reload_checkpoint': None,
    'valid_max_iter': 10,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': '../checkpoints/'
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 64,
    'cpu_workers': 4,
    'reload_checkpoint': '../checkpoints/crnn_092000_loss0.8285654664039612.pkl',
    'decode_method': 'beam_search',
    'beam_size': 10,
}
evaluate_config.update(common_config)
